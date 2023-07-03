# Secret management

Some models need access to APIs, databases, AWS resources, or other secured information. With Truss, you can securely reference access keys, API tokens, passwords, secrets, and more. Keep this sensitive information out of your Git repository and use the same best practices for ML models as production services.

### Accessing secrets in models

This example is based on the [GFP-GAN example Truss](https://github.com/basetenlabs/truss/tree/main/examples/gfpgan) in the main repository. That model needs access to an S3 bucket on AWS, requiring an access key id, access key secret, AWS region, and bucket name. These values are handled as secrets.

{% hint style="danger" %}
Never commit secret values to source control.
{% endhint %}

Start by defining a list of secrets in the `config.yaml` file. YAML is a key-value store, but you don't want to **ever** actually store the secret values in this file, even when running the Truss locally. Instead, set default values for the secrets, like `null`, and can leave comments about what kind of value to expect. Default values can also be fallback values that show format or type, like an empty string, the number 0, or an example UUID.

```yaml
secrets:
  gfpgan_aws_access_key_id: null
  gfpgan_aws_secret_access_key: null
  gfpgan_aws_region: null # e.g. us-east-1
  gfpgan_aws_bucket: null
```

{% hint style="warning" %}
YAML syntax can be a bit non-obvious when dealing with empty dictionaries. You may notice the following in the default Truss config file:

```yaml
secrets: {}
```

When you fill them in with values, dictionaries should look like this:

```yaml
secrets:
  key1: default_value1
  key2: default_value2
```
{% endhint %}

Then, you can access the secrets in the `model/model.py` file by referencing them as kwargs in the init function.

```python
class Model:

    def __init__(self, **kwargs):
        self._config = kwargs.get("config")
        secrets = kwargs.get("secrets")
        # Use secrets via dictionary in any function within the Model class
        self.s3_config = (
            {
                "aws_access_key_id": secrets["gfpgan_aws_access_key_id"],
                "aws_secret_access_key": secrets["gfpgan_aws_secret_access_key"],
                "aws_region": secrets["gfpgan_aws_region"],
            }
        )
        self.s3_bucket = (secrets["gfpgan_aws_bucket"])

    def load(self):
        ...

    def predict(self, model_input):
        ...
```

## Setting secrets locally

When you run Truss locally, it creates a folder at `~/.truss` to store configuration and temporary files. This folder may or may not contain a file `config.yaml`. If that file doesn't exist, make it with:

```sh
touch ~/.truss/config.yaml
```

Then add the same secret names that you use in your Truss config in this local config, but actually set the secret value in `~/.truss/config.yaml`.

**/path/to/your/truss/config.yaml:**

```yaml
secrets:
  MY_API_TOKEN: null
```

**~/.truss/config.yaml:**

```yaml
secrets:
  MY_API_TOKEN: "abcd1234.qwerty"
```

## Setting secrets in production

When running your model in production (or a production-like environment such as staging), you have two options: environment variables and mounted secrets.

### Environment variables

To set a secret value in production, you can set an environment variable and reference its value in your Truss. To avoid namespace conflicts, environment variable names must be prefixed with `TRUSS_SECRET_`. For example:

**config.yaml:**

```yaml
secrets:
  MY_API_TOKEN: null
```

**Environment variable:**

```sh
TRUSS_SECRET_MY_API_TOKEN="abcd1234.qwerty"
```

### Mounting secrets

You can also mount secrets in Kubernetes by following [this documentation](https://kubernetes.io/docs/concepts/configuration/secret/).

Mounted secrets should not use the `TRUSS_SECRET_` prefix as there is no need to avoid namespace conflicts.

## Deploying with secrets

If you're deploying your model to Baseten, set `is_trusted=True` in the `deploy()` command to enable your model to access secrets:

```python
import baseten
basten.deploy(
    my_truss,
    model_name="My Model",
    is_trusted=True
)
```

Secrets can be securely stored in your Baseten organization by following [this documentation](https://docs.baseten.co/settings/secrets).

{% hint style="warning" %}
Baseten mounts secrets, so do not use the `TRUSS_SECRET_` prefix when setting secret names.
{% endhint %}

If you're deploying your model to another platform, reference that platform's documentation for setting environment variables or mounting a secrets volume.
