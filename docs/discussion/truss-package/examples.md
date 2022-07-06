# Examples

Examples can be provided in a Truss in a yaml file, by default called
`examples.yaml`. Examples are great documentation. Truss even provides a
command to run any or all examples of the model.

An example file is just a mapping of request dictionaries by example names.

Running all examples:
```bash
truss run-example
```

Running specific example:
```bash
truss run-example --name example_name
```


Example examples file:
#### **`examples.yaml`**
```examples.yaml
cat_example:
  instances:
    - image_url: 'https://source.unsplash.com/gKXKBY-C-Dk/300x300'
      labels:
        - small cat
        - not cat
        - big cat
```
