# Truss Architecture

At a high level this tool builds on top of existing frameworks. Out of the box
we support the packages `sklearn`, `keras`, and `pytorch` and their models. Our
serving infrastructure is based on `kfserving` with some opinions. However the
architecture of Truss is suggestive and not prescriptive. The idea of the
package is that one can make changes to the structure locally and test it.
Significant divergence from the suggestions may cause incompatibility with
BaseTen's serving environment, but might be required for other use cases.
