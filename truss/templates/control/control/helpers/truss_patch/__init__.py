# Truss patch is meant for the machinery of patching a Truss itself.
# This is completely different from the monkey patches we have for
# various ML libraries.
# Truss may be patched in different forms:
#   - Patching the truss directory itself to updae contents
#   - Patching the truss as laid out on the generated docker image
#     (which may be different from the truss dir structure)
#   - Patching the container environment, e.g. installing/removing pip and system packages
# Right now we bundle this folder in both truss and in live_reload images. This
# is a good candidate for extracting out into a smaller library.
# Protocol of usage for now is that Truss library can use the code here,
# but the code here should use Truss library very sparingly, perhaps just TrussConfig.
# This because control server can only use Truss library code via published
# versions, so getting changes in Truss server requires Truss code to be published
# and the control server's requirements.txt updated.
