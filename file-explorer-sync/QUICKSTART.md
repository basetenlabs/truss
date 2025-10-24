# Quick Start Guide

## What This Extension Does

The **File Explorer Sync** extension automatically expands and collapses directories in your file explorer as you navigate between files in the editor, keeping your file tree organized and in sync with your current work.

## Key Features

### 🎯 Automatic Syncing
- When you open a file, the file explorer automatically expands to show that file
- No manual navigation needed in the file tree

### 👁️ Status Bar Toggle
- Click the eye icon in the status bar to quickly enable/disable syncing
- Visual indicator shows current state

### 🎛️ Customizable Behavior
- Choose whether to collapse folders when files are closed
- Keep focus on editor or switch to explorer
- Configure reveal depth for nested directories

## How to Use

### Method 1: Launch in Development Mode (Fastest)

1. Open VS Code/Cursor in the extension folder:
   ```bash
   cd /workspace/file-explorer-sync
   code .
   ```

2. Press **F5** to launch Extension Development Host

3. In the new window, open any workspace folder

4. Start opening files - watch the explorer sync automatically!

### Method 2: Install Locally

1. Copy the extension to your extensions folder:
   ```bash
   # For VS Code
   cp -r /workspace/file-explorer-sync ~/.vscode/extensions/
   
   # For Cursor
   cp -r /workspace/file-explorer-sync ~/.cursor/extensions/
   ```

2. Restart VS Code/Cursor

3. Open any workspace and start coding

## Testing the Extension

1. **Open a deeply nested file**
   - Navigate to any file several folders deep
   - The explorer should automatically expand to show it

2. **Switch between files**
   - Open multiple files in different directories
   - The explorer should follow your navigation

3. **Toggle the feature**
   - Click the status bar icon (👁 Explorer Sync)
   - Open another file - it shouldn't sync
   - Click again to re-enable

4. **Test collapse on close** (optional)
   - Enable in settings: `"fileExplorerSync.collapseOnClose": true`
   - Close all files in a directory
   - The directory should collapse

## Commands

Access via Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`):

- **Toggle File Explorer Sync** - Enable/disable the feature
- **Sync File Explorer Now** - Manually sync with current file

## Configuration Examples

### Minimal Configuration (Default)
```json
{
  "fileExplorerSync.enabled": true
}
```

### Full-Featured Configuration
```json
{
  "fileExplorerSync.enabled": true,
  "fileExplorerSync.collapseOnClose": true,
  "fileExplorerSync.focusExplorer": false,
  "fileExplorerSync.revealDepth": -1
}
```

### Focus Explorer on Sync
```json
{
  "fileExplorerSync.enabled": true,
  "fileExplorerSync.focusExplorer": true
}
```

## Troubleshooting

### Extension not activating?
- Check that you're using VS Code 1.80.0 or higher
- Look for errors in Output > Extension Host

### Not syncing?
- Check the status bar icon - is it enabled?
- Try the "Sync File Explorer Now" command
- Make sure you're opening files with `file://` scheme (not virtual files)

### Collapsing not working?
- Enable `fileExplorerSync.collapseOnClose` in settings
- This feature is experimental and may not work perfectly with all folder structures

## Next Steps

1. Try it with your real projects
2. Adjust settings to match your workflow
3. Share feedback for improvements!

## Development

To modify the extension:

1. Edit `extension.js`
2. Reload the Extension Development Host (Ctrl+R or Cmd+R)
3. Test your changes

## Support

For issues or questions:
- Check the README.md for detailed documentation
- Review INSTALL.md for installation help
- Check the code in extension.js for implementation details
