import fs from "node:fs"
import path from "node:path"
import process from "node:process"

const workspace = process.cwd()
const rootPackagePath = path.join(workspace, "package.json")
const npmDirectory = path.join(workspace, "npm")
const rootPackage = JSON.parse(fs.readFileSync(rootPackagePath, "utf8"))

const platformPackages = fs
  .readdirSync(npmDirectory, { withFileTypes: true })
  .filter((entry) => entry.isDirectory())
  .map((entry) => {
    const directory = path.join(npmDirectory, entry.name)
    const packagePath = path.join(directory, "package.json")
    const packageJson = JSON.parse(fs.readFileSync(packagePath, "utf8"))
    const binaryPath = path.join(directory, packageJson.main)
    return { binaryPath, directory, packageJson }
  })

const preparedPackages = platformPackages.filter(({ binaryPath }) =>
  fs.existsSync(binaryPath)
)
const skippedPackages = platformPackages.filter(
  ({ binaryPath }) => !fs.existsSync(binaryPath)
)

if (preparedPackages.length === 0) {
  throw new Error("No platform packages contain native binaries")
}

for (const { binaryPath, packageJson } of preparedPackages) {
  if (!packageJson.name || !packageJson.main) {
    throw new Error(`Invalid platform package for ${binaryPath}`)
  }
  if (packageJson.version !== rootPackage.version) {
    throw new Error(
      `${packageJson.name} has version ${packageJson.version}; expected ${rootPackage.version}`
    )
  }
  if (fs.statSync(binaryPath).size === 0) {
    throw new Error(`${binaryPath} is empty`)
  }
}

const optionalDependencies = Object.fromEntries(
  preparedPackages
    .map(({ packageJson }) => [packageJson.name, packageJson.version])
    .sort(([left], [right]) => left.localeCompare(right))
)

if (Object.keys(optionalDependencies).length !== preparedPackages.length) {
  throw new Error("Platform package names must be unique")
}

rootPackage.optionalDependencies = optionalDependencies
fs.writeFileSync(rootPackagePath, `${JSON.stringify(rootPackage, null, 2)}\n`)

for (const { packageJson } of skippedPackages) {
  console.log(`Skipping ${packageJson.name}: native binary was not built`)
}
console.log(
  `Prepared ${preparedPackages.length} platform packages for ${rootPackage.name}@${rootPackage.version}`
)
