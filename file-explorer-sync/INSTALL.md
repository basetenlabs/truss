# Installation Guide

## Method 1: Install from VSIX (Recommended for Local Use)

1. Package the extension:
   ```bash
   cd file-explorer-sync
   npm install -g vsce
   vsce package
   ```

2. Install the generated `.vsix` file:
   - Open VS Code/Cursor
   - Go to Extensions view (`Ctrl+Shift+X` or `Cmd+Shift+X`)
   - Click the `...` menu at the top
   - Select "Install from VSIX..."
   - Choose the generated `.vsix` file

## Method 2: Development Mode

1. Open the extension folder in VS Code/Cursor:
   ```bash
   cd file-explorer-sync
   code .
   ```

2. Press `F5` to launch Extension Development Host

3. The extension will be active in the new window

## Method 3: Copy to Extensions Folder

1. Copy the entire `file-explorer-sync` folder to your VS Code extensions directory:
   - **Windows**: `%USERPROFILE%\.vscode\extensions\`
   - **macOS**: `~/.vscode/extensions/`
   - **Linux**: `~/.vscode/extensions/`

2. For Cursor, use:
   - **Windows**: `%USERPROFILE%\.cursor\extensions\`
   - **macOS**: `~/.cursor/extensions/`
   - **Linux**: `~/.cursor/extensions/`

3. Restart VS Code/Cursor

## Verification

After installation, you should see:
- A status bar item showing "👁 Explorer Sync" in the bottom right
- Commands available in Command Palette (`Ctrl+Shift+P`):
  - "Toggle File Explorer Sync"
  - "Sync File Explorer Now"

## Configuration

Access settings via:
- `File > Preferences > Settings` (or `Ctrl+,`)
- Search for "File Explorer Sync"
- Adjust settings as needed

## Troubleshooting

If the extension doesn't activate:
1. Check the Output panel (`View > Output`) and select "Extension Host" from dropdown
2. Look for any error messages related to "file-explorer-sync"
3. Try reloading the window: `Developer: Reload Window` from Command Palette
