use super::hf_metadata::HfError;

pub fn normalize_hash(hash: &str) -> String {
    // remove characters that break linux flat folder names, such as
    // slashes, colons, etc.

    hash.replace(['/', ':', '\\', '*', '?', '"', '<', '>', '|'], "_")
        .replace(' ', "_")
}

/// Simple glob pattern matching
pub fn glob_match(pattern: &str, text: &str) -> bool {
    if pattern == "*" {
        return true;
    }

    if pattern.contains('*') {
        let parts: Vec<&str> = pattern.split('*').collect();
        let mut text_pos = 0;

        for (i, part) in parts.iter().enumerate() {
            if part.is_empty() {
                continue;
            }

            if i == 0 {
                // First part must match from beginning
                if !text[text_pos..].starts_with(part) {
                    return false;
                }
                text_pos += part.len();
            } else if i == parts.len() - 1 {
                // Last part must match at end
                return text[text_pos..].ends_with(part);
            } else {
                // Middle parts must exist somewhere
                if let Some(pos) = text[text_pos..].find(part) {
                    text_pos += pos + part.len();
                } else {
                    return false;
                }
            }
        }
        true
    } else {
        text == pattern
    }
}

/// Check if file should be ignored based on patterns
pub fn should_ignore_file(
    file_path: &str,
    allow_patterns: Option<&[String]>,
    ignore_patterns: Option<&[String]>,
) -> bool {
    // If there are ignore patterns and this file matches any, ignore it
    if let Some(ignore) = ignore_patterns {
        for pattern in ignore {
            if glob_match(pattern, file_path) {
                return true;
            }
        }
    }

    // If there are allow patterns, file must match at least one
    if let Some(allow) = allow_patterns {
        for pattern in allow {
            if glob_match(pattern, file_path) {
                return false; // Found a match, don't ignore
            }
        }
        return true; // No match found, ignore
    }

    false // No patterns or no ignore match
}

/// Filter repository files based on patterns
pub fn filter_repo_files(
    files: Vec<String>,
    allow_patterns: Option<&[String]>,
    ignore_patterns: Option<&[String]>,
) -> Result<Vec<String>, HfError> {
    let total_files = files.len();
    let filtered_files = files
        .into_iter()
        .filter(|file| !should_ignore_file(file, allow_patterns, ignore_patterns))
        .collect::<Vec<_>>();

    if filtered_files.is_empty() {
        return Err(HfError::InvalidMetadata(format!(
            "No files matched weights filters. Listed {total_files} file(s), \
            allow_patterns={:?}, ignore_patterns={:?}",
            allow_patterns, ignore_patterns
        )));
    }

    Ok(filtered_files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glob_match() {
        assert!(glob_match("*.txt", "file.txt"));
        assert!(glob_match("*.json", "config.json"));
        assert!(!glob_match("*.txt", "file.json"));
        assert!(glob_match("*", "anything"));
        assert!(glob_match("prefix*", "prefix_file.txt"));
        assert!(!glob_match("prefix*", "other_file.txt"));

        // Test lock file pattern specifically
        assert!(glob_match("*.lock", "tokenizer_config.json.lock"));
        assert!(glob_match("*.lock", "file.lock"));
        assert!(!glob_match("*.lock", "file.txt"));
    }

    #[test]
    fn test_should_ignore_file() {
        let allow_patterns = vec!["*.safetensors".to_string(), "*.json".to_string()];
        let ignore_patterns = vec!["*.md".to_string()];

        // Should allow safetensors files
        assert!(!should_ignore_file(
            "model.safetensors",
            Some(&allow_patterns),
            Some(&ignore_patterns)
        ));

        // Should ignore md files
        assert!(should_ignore_file(
            "README.md",
            Some(&allow_patterns),
            Some(&ignore_patterns)
        ));

        // Should ignore files not in allow patterns
        assert!(should_ignore_file(
            "model.txt",
            Some(&allow_patterns),
            Some(&ignore_patterns)
        ));

        // Should allow when no patterns specified
        assert!(!should_ignore_file("any_file.txt", None, None));
    }

    #[test]
    fn test_filter_repo_files_errors_when_allow_patterns_match_nothing() {
        let files = vec!["model.bin".to_string(), "README.md".to_string()];
        let allow_patterns = vec!["*.safetensors".to_string()];

        let err = filter_repo_files(files, Some(&allow_patterns), None).unwrap_err();

        assert!(err.to_string().contains("No files matched weights filters"));
        assert!(err.to_string().contains("Listed 2 file(s)"));
    }

    #[test]
    fn test_filter_repo_files_errors_when_repo_listing_is_empty() {
        let err = filter_repo_files(vec![], None, None).unwrap_err();

        assert!(err.to_string().contains("No files matched weights filters"));
        assert!(err.to_string().contains("Listed 0 file(s)"));
    }
}
