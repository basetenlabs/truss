# Consistency Summary: Truss and Chains Push Behavior

This document summarizes the consistent behavior across both `truss push` and `truss chains push` commands.

## 🎯 **Core Semantics (Consistent Across Both)**

### **Default Behavior**
- **Command**: `truss push` or `truss chains push` (no flags)
- **Result**: Creates **published deployment**
- **Rationale**: Published deployments are the most common use case

### **Development Behavior**
- **Command**: `truss push --watch` or `truss chains push --watch`
- **Result**: Creates **development deployment** with hot reload support
- **Features**:
  - Automatic log streaming (like `--tail`)
  - Live patching mode after logs complete
  - Clear user feedback throughout process

### **Environment-Based Deployment**
- **Command**: `truss push --environment {env_name}` or `truss chains push --environment {env_name}`
- **Result**: Creates deployment and initiates promotion to specified environment
- **Examples**:
  - `--environment production` → Production deployment
  - `--environment staging` → Staging deployment
  - `--environment dev` → Development deployment

## 🚫 **Deprecated Flags (Consistent Across Both)**

### **--publish Flag**
- **Status**: `[DEPRECATED]`
- **Reason**: Now the default behavior
- **Alternative**: Use no flags (default) or `--watch` for development
- **Removal**: Will be removed in the following release
- **Warning**: Shows deprecation warning when used

### **--promote Flag**
- **Status**: `[DEPRECATED]`
- **Reason**: Less explicit than environment-based approach
- **Alternative**: Use `--environment production` or `--environment {env_name}`
- **Removal**: Will be removed in the following release
- **Warning**: Shows deprecation warning when used

## ✅ **Validation & Error Handling (Consistent Across Both)**

### **Flag Conflicts**
- `--watch` + `--publish`: Error - cannot use both
- `--watch` + `--promote`: Error - cannot use both
- `--publish` + `--promote`: Allowed (both create published deployment)

### **Transport Restrictions**
- **gRPC Transport**: Cannot be used with `--watch` (development deployments)
- **Error Message**: Clear guidance to use published deployment instead

### **TRT-LLM Restrictions**
- **Development Mode**: Not supported for TRT-LLM build flow
- **Error Message**: Clear guidance to use published deployment

## 📱 **User Experience (Consistent Across Both)**

### **Help Text**
- Both commands show `[DEPRECATED]` markers for deprecated flags
- Clear guidance on alternatives to use
- Consistent formatting and messaging

### **Runtime Warnings**
- Console warnings with yellow styling for deprecated flags
- Python `DeprecationWarning` for programmatic users
- Clear migration paths provided

### **Error Messages**
- Consistent language across both commands
- References to `--environment {env_name}` instead of deprecated flags
- Helpful guidance for resolution

## 🧪 **Test Coverage**

### **Comprehensive Test Suite**
- **8 tests** covering all scenarios
- **Consistency verification** across both commands
- **Deprecation warning testing** for both flags
- **Default behavior verification** for both commands
- **Error handling validation** for edge cases

### **Test Categories**
1. **Default Behavior**: Both default to published deployments
2. **Watch Behavior**: Both support development with `--watch`
3. **Flag Conflicts**: Proper validation of conflicting flags
4. **Transport Restrictions**: gRPC transport handling
5. **Deprecation Warnings**: Both flags show proper warnings
6. **Help Text**: Consistent deprecation markers
7. **Error Messages**: Consistent language and guidance

## 🔄 **Migration Path**

### **For Users**
1. **Current**: `truss push --publish` → Shows warning, works as before
2. **Recommended**: `truss push` → Default published deployment
3. **Development**: `truss push --watch` → Development with live patching
4. **Production**: `truss push --environment production` → Explicit production

### **Future Removal**
- Both `--publish` and `--promote` will be removed in the following release
- Users encouraged to adopt new patterns before removal
- Backward compatibility maintained until removal

## 🎉 **Benefits of Consistency**

1. **Predictable Behavior**: Same semantics across both commands
2. **Simplified Mental Model**: One set of rules to learn
3. **Better UX**: Clear defaults and explicit development mode
4. **Future-Proof**: Environment-based approach is more flexible
5. **Clean Migration**: Deprecation warnings guide users smoothly

## 📊 **Behavior Matrix**

| Command | Flags | Deployment Type | Status |
|---------|-------|----------------|--------|
| `truss push` | none | **Published** | ✅ Default |
| `truss chains push` | none | **Published** | ✅ Default |
| `truss push --watch` | `--watch` | **Development** | ✅ Explicit |
| `truss chains push --watch` | `--watch` | **Development** | ✅ Explicit |
| `truss push --environment prod` | `--environment` | **Published** | ✅ Explicit |
| `truss chains push --environment prod` | `--environment` | **Published** | ✅ Explicit |
| `truss push --publish` | `--publish` | Published | ⚠️ Deprecated |
| `truss chains push --publish` | `--publish` | Published | ⚠️ Deprecated |
| `truss push --promote` | `--promote` | Published | ⚠️ Deprecated |
| `truss chains push --promote` | `--promote` | Published | ⚠️ Deprecated |

This consistency ensures that users have the same experience regardless of whether they're working with individual trusses or chains, while providing a clear migration path to the new recommended patterns.
