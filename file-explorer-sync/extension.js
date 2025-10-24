const vscode = require('vscode');

let isEnabled = true;
let openFiles = new Set();
let statusBarItem;

function activate(context) {
    console.log('File Explorer Sync extension is now active');

    statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBarItem.command = 'fileExplorerSync.toggle';
    updateStatusBar();
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);

    const config = vscode.workspace.getConfiguration('fileExplorerSync');
    isEnabled = config.get('enabled', true);

    context.subscriptions.push(
        vscode.window.onDidChangeActiveTextEditor(editor => {
            if (isEnabled && editor && editor.document && editor.document.uri.scheme === 'file') {
                syncFileExplorer(editor.document.uri);
            }
        })
    );

    context.subscriptions.push(
        vscode.workspace.onDidOpenTextDocument(document => {
            if (document && document.uri.scheme === 'file') {
                openFiles.add(document.uri.fsPath);
            }
        })
    );

    context.subscriptions.push(
        vscode.workspace.onDidCloseTextDocument(document => {
            if (document && document.uri.scheme === 'file') {
                openFiles.delete(document.uri.fsPath);
                
                const config = vscode.workspace.getConfiguration('fileExplorerSync');
                if (isEnabled && config.get('collapseOnClose', false)) {
                    collapseIfEmpty(document.uri);
                }
            }
        })
    );

    context.subscriptions.push(
        vscode.workspace.onDidChangeConfiguration(e => {
            if (e.affectsConfiguration('fileExplorerSync.enabled')) {
                const config = vscode.workspace.getConfiguration('fileExplorerSync');
                isEnabled = config.get('enabled', true);
                updateStatusBar();
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('fileExplorerSync.toggle', () => {
            isEnabled = !isEnabled;
            const config = vscode.workspace.getConfiguration('fileExplorerSync');
            config.update('enabled', isEnabled, vscode.ConfigurationTarget.Global);
            updateStatusBar();
            vscode.window.showInformationMessage(
                `File Explorer Sync ${isEnabled ? 'enabled' : 'disabled'}`
            );
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('fileExplorerSync.syncNow', () => {
            const editor = vscode.window.activeTextEditor;
            if (editor && editor.document && editor.document.uri.scheme === 'file') {
                syncFileExplorer(editor.document.uri, true);
                vscode.window.showInformationMessage('File explorer synced');
            } else {
                vscode.window.showWarningMessage('No active file to sync');
            }
        })
    );

    vscode.workspace.textDocuments.forEach(doc => {
        if (doc.uri.scheme === 'file') {
            openFiles.add(doc.uri.fsPath);
        }
    });

    const activeEditor = vscode.window.activeTextEditor;
    if (isEnabled && activeEditor && activeEditor.document.uri.scheme === 'file') {
        syncFileExplorer(activeEditor.document.uri);
    }
}

function updateStatusBar() {
    if (isEnabled) {
        statusBarItem.text = '$(eye) Explorer Sync';
        statusBarItem.tooltip = 'File Explorer Sync is enabled (click to toggle)';
        statusBarItem.backgroundColor = undefined;
    } else {
        statusBarItem.text = '$(eye-closed) Explorer Sync';
        statusBarItem.tooltip = 'File Explorer Sync is disabled (click to toggle)';
        statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
    }
}

async function syncFileExplorer(uri, force = false) {
    if (!isEnabled && !force) {
        return;
    }

    const config = vscode.workspace.getConfiguration('fileExplorerSync');
    const focusExplorer = config.get('focusExplorer', false);

    try {
        await vscode.commands.executeCommand('revealInExplorer', uri);
        
        if (!focusExplorer) {
            await vscode.commands.executeCommand('workbench.action.focusActiveEditorGroup');
        }
    } catch (error) {
        console.error('Error syncing file explorer:', error);
    }
}

async function collapseIfEmpty(closedUri) {
    const path = require('path');
    const fs = require('fs');
    
    try {
        const dirPath = path.dirname(closedUri.fsPath);
        
        const hasOpenFilesInDir = Array.from(openFiles).some(filePath => {
            return path.dirname(filePath) === dirPath;
        });

        if (!hasOpenFilesInDir) {
            await vscode.commands.executeCommand('list.collapseAllToFocus');
        }
    } catch (error) {
        console.error('Error collapsing directory:', error);
    }
}

function deactivate() {
    if (statusBarItem) {
        statusBarItem.dispose();
    }
}

module.exports = {
    activate,
    deactivate
};
