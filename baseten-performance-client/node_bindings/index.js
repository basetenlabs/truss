try {
  module.exports = require('./baseten-performance-client-node.linux-x64-gnu.node');
} catch (e) {
  // Try alternative names
  try {
    module.exports = require('./baseten-performance-client-node.node');
  } catch (e2) {
    try {
      module.exports = require('./index.node');
    } catch (e3) {
      throw new Error(`Could not load native bindings: ${e.message}, ${e2.message}, ${e3.message}`);
    }
  }
}
