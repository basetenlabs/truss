# File Explorer Sync

Automatically expands and collapses directories in the file explorer as you open and close files in the editor.

## Features

- **Automatic Syncing**: Automatically reveals files in the explorer when you open them
- **Smart Collapsing**: Optionally collapse directories when all files in them are closed
- **Configurable**: Multiple settings to customize the behavior
- **Status Bar Integration**: Quick toggle via status bar icon
- **Non-intrusive**: Keeps focus on the editor by default

## Commands

- `Toggle File Explorer Sync`: Enable/disable the automatic syncing
- `Sync File Explorer Now`: Manually sync the explorer with the current file

## Configuration

| Setting | Description | Default |
|---------|-------------|---------|
| `fileExplorerSync.enabled` | Enable automatic file explorer syncing | `true` |
| `fileExplorerSync.collapseOnClose` | Collapse directories when all files in them are closed | `false` |
| `fileExplorerSync.revealDepth` | Maximum depth to reveal in tree (-1 for unlimited) | `-1` |
| `fileExplorerSync.focusExplorer` | Focus the file explorer when syncing | `false` |

## Usage

1. Install the extension
2. The extension activates automatically
3. Open files in the editor and watch the file explorer sync automatically
4. Click the status bar icon (👁 Explorer Sync) to toggle the feature on/off
5. Configure settings in VS Code preferences under "File Explorer Sync"

## Example Configuration

Add to your `settings.json`:

```json
{
  "fileExplorerSync.enabled": true,
  "fileExplorerSync.collapseOnClose": true,
  "fileExplorerSync.focusExplorer": false
}
```

## Requirements

- VS Code 1.80.0 or higher

## Known Limitations

- The collapse feature may not work perfectly with nested directories
- Some edge cases with multiple workspace folders might need manual sync

## Development

To test this extension locally:

1. Open this folder in VS Code
2. Press `F5` to launch Extension Development Host
3. Open a folder and test the functionality

## License

MIT
