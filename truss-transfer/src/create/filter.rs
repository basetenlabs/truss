use super::hf_metadata::HfError;

/// Filter repository files based on patterns
pub fn filter_repo_files(
    files: Vec<String>,
    allow_patterns: Option<&[String]>,
    ignore_patterns: Option<&[String]>,
) -> Result<Vec<String>, HfError> {
    let mut filtered_files = files;

    // Apply allow patterns (if specified, only keep files that match)
    if let Some(patterns) = allow_patterns {
        filtered_files = filtered_files
            .into_iter()
            .filter(|file| {
                patterns.iter().any(|pattern| {
                    glob::Pattern::new(pattern)
                        .map_err(|e| HfError::Pattern(e.to_string()))
                        .map(|p| p.matches(file))
                        .unwrap_or(false)
                })
            })
            .collect();
    }

    // Apply ignore patterns (remove files that match)
    if let Some(patterns) = ignore_patterns {
        filtered_files = filtered_files
            .into_iter()
            .filter(|file| {
                !patterns.iter().any(|pattern| {
                    glob::Pattern::new(pattern)
                        .map_err(|e| HfError::Pattern(e.to_string()))
                        .map(|p| p.matches(file))
                        .unwrap_or(false)
                })
            })
            .collect();
    }

    Ok(filtered_files)
}
