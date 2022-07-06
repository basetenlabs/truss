# Data directory

Models often need access to the model binary, reference tables and other data.
These can be placed under a directory named `data`. This data is made available
to the model code at serving time as an initialization argument called
`data_dir` of type pathlib.Path.

The data directory can have sub-directories, any level deep. Symlinks are not
supported.
