#!/usr/bin/env node

/**
 * Install platform-specific dependencies for packaging VS Code extension
 *
 * This script installs only the platform-specific native modules for the
 * current platform, so the VSIX package only includes binaries for the
 * target platform.
 *
 * npm by default respects the "os" and "cpu" fields in package.json and won't
 * install packages for other platforms. This script bypasses that by downloading
 * the packages directly from the npm registry and creating proper package.json
 * entries so vsce can validate them.
 */

const fs = require('fs');
const path = require('path');
const https = require('https');

// Configuration: Define which package scopes to include as platform-specific
const PLATFORM_PACKAGE_CONFIGS = [
  {
    scope: '@lancedb',
    prefix: 'lancedb-',
    description: 'LanceDB',
    expectedCount: 1,
    sources: ['dependencies', 'optionalDependencies']
  },
  {
    scope: '@img',
    prefix: 'sharp-',
    description: 'Sharp',
    expectedCount: 1,
    sources: ['dependencies', 'optionalDependencies']
  },
  {
    scope: '@img',
    prefix: 'sharp-libvips-',
    description: 'Sharp libvips',
    expectedCount: 1,
    sources: ['dependencies', 'optionalDependencies']
  }
];

/**
 * Parse target platform from argument or environment variable
 * Format: {platform}-{arch} (e.g., "linux-x64", "darwin-arm64", "win32-x64")
 */
function getTargetPlatform() {
  // Check command line arguments first
  const args = process.argv.slice(2);
  let target = args.find(arg => !arg.startsWith('--'));

  // Check environment variable if not provided as argument
  if (!target) {
    target = process.env.VSCE_TARGET || process.env.TARGET;
  }

  if (!target) {
    throw new Error(
      'Target platform is required. ' +
      'Usage: node install-platform-deps.js <target> ' +
      'or set VSCE_TARGET environment variable. ' +
      'Examples: linux-x64, darwin-arm64, win32-x64'
    );
  }

  // Parse target format: {platform}-{arch}
  const parts = target.split('-');
  if (parts.length < 2) {
    throw new Error(`Invalid target format: ${target}. Expected format: {platform}-{arch} (e.g., linux-x64)`);
  }

  const platformName = parts[0]; // 'darwin', 'linux', 'win32'
  const archName = parts[1]; // 'arm64', 'x64'

  // Validate platform
  const validPlatforms = ['darwin', 'linux', 'win32'];
  if (!validPlatforms.includes(platformName)) {
    throw new Error(`Invalid platform: ${platformName}. Must be one of: ${validPlatforms.join(', ')}`);
  }

  // Validate architecture
  const validArchs = ['arm64', 'x64'];
  if (!validArchs.includes(archName)) {
    throw new Error(`Invalid architecture: ${archName}. Must be one of: ${validArchs.join(', ')}`);
  }

  // Generate patterns that match packages for this platform
  // LanceDB uses variants like: linux-x64-gnu, win32-x64-msvc
  const patterns = [`${platformName}-${archName}`];

  if (platformName === 'linux') {
    patterns.push(`${platformName}-${archName}-gnu`);
  } else if (platformName === 'win32') {
    patterns.push(`${platformName}-${archName}-msvc`);
  }

  return {
    platform: platformName,
    arch: archName,
    target: `${platformName}-${archName}`,
    patterns
  };
}

/**
 * Check if a package name matches the target platform
 */
function matchesTargetPlatform(packageName, platformPatterns) {
  // Extract the platform identifier from package name
  // e.g., "@img/sharp-darwin-arm64" -> "darwin-arm64"
  // e.g., "@lancedb/lancedb-linux-x64-gnu" -> "linux-x64-gnu"
  const parts = packageName.split('/');
  if (parts.length !== 2) return false;

  const packagePart = parts[1];

  // Check if any platform pattern matches
  // A pattern matches if:
  // 1. The package name starts with the pattern (e.g., "darwin-arm64" matches "darwin-arm64")
  // 2. The package name contains the pattern followed by a hyphen (e.g., "linux-x64" matches "linux-x64-gnu")
  return platformPatterns.some(pattern => {
    // Check if package name contains the pattern
    const patternIndex = packagePart.indexOf(pattern);
    if (patternIndex === -1) return false;

    // Check if pattern is at the end or followed by a hyphen (for variants like -gnu, -msvc)
    const afterPattern = packagePart.substring(patternIndex + pattern.length);
    return afterPattern === '' || afterPattern.startsWith('-');
  });
}

// Read package.json
const packageJsonPath = path.join(__dirname, '..', 'package.json');
const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));

// Get target platform
const targetPlatform = getTargetPlatform();
console.log(`Installing platform-specific dependencies for ${targetPlatform.target}...\n`);

// Collect platform-specific packages based on configuration
const platformDeps = [];

for (const config of PLATFORM_PACKAGE_CONFIGS) {
  for (const source of config.sources) {
    const deps = packageJson[source] || {};
    const filtered = Object.entries(deps)
      .filter(([name]) => {
        // First check if it's a platform-specific package
        if (!name.startsWith(`${config.scope}/${config.prefix}`)) {
          return false;
        }
        // Then check if it matches target platform
        return matchesTargetPlatform(name, targetPlatform.patterns);
      })
      .map(([name, version]) => ({
        name,
        version: version.replace(/^[\^~]/, ''),
        config
      }));
    platformDeps.push(...filtered);
  }
}

if (platformDeps.length === 0) {
  console.log(`No platform-specific dependencies found for target platform ${targetPlatform.target}.`);
  process.exit(0);
}

console.log('Target platform-specific binaries:');
platformDeps.forEach(({name, version}) => console.log(`  - ${name}@${version}`));
console.log();

async function downloadAndExtract(packageName, version, targetDir) {
  const tarball = `https://registry.npmjs.org/${packageName}/-/${packageName.split('/')[1]}-${version}.tgz`;

  console.log(`Downloading ${packageName}@${version}...`);

  return new Promise((resolve, reject) => {
    https.get(tarball, (response) => {
      if (response.statusCode === 302 || response.statusCode === 301) {
        // Follow redirect
        https.get(response.headers.location, (redirectResponse) => {
          if (redirectResponse.statusCode !== 200) {
            reject(new Error(`Failed to download: ${redirectResponse.statusCode}`));
            return;
          }

          const tar = require('child_process').spawn('tar', ['xz', '-C', targetDir, '--strip-components=1'], {
            stdio: ['pipe', 'inherit', 'inherit']
          });

          redirectResponse.pipe(tar.stdin);
          tar.on('close', (code) => {
            if (code === 0) {
              resolve();
            } else {
              reject(new Error(`tar extraction failed with code ${code}`));
            }
          });
        });
      } else if (response.statusCode === 200) {
        const tar = require('child_process').spawn('tar', ['xz', '-C', targetDir, '--strip-components=1'], {
          stdio: ['pipe', 'inherit', 'inherit']
        });

        response.pipe(tar.stdin);
        tar.on('close', (code) => {
          if (code === 0) {
            resolve();
          } else {
            reject(new Error(`tar extraction failed with code ${code}`));
          }
        });
      } else {
        reject(new Error(`Failed to download: ${response.statusCode}`));
      }
    }).on('error', reject);
  });
}

async function removePlatformRestrictions(packageDir) {
  // Remove os/cpu restrictions from package.json to avoid npm validation errors
  const pkgPath = path.join(packageDir, 'package.json');
  if (fs.existsSync(pkgPath)) {
    const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));
    delete pkg.os;
    delete pkg.cpu;
    fs.writeFileSync(pkgPath, JSON.stringify(pkg, null, 2));
  }
}

/**
 * Remove platform-specific packages that don't match the target platform
 * This ensures each package operation only includes dependencies for the target platform
 */
function cleanupOtherPlatformPackages(targetPlatform) {
  console.log(`Cleaning up platform-specific packages not matching ${targetPlatform.target}...\n`);

  for (const config of PLATFORM_PACKAGE_CONFIGS) {
    const scopeDir = path.join(__dirname, '..', 'node_modules', config.scope);

    if (!fs.existsSync(scopeDir)) {
      continue;
    }

    const installedPackages = fs.readdirSync(scopeDir)
      .filter(dir => dir.startsWith(config.prefix));

    for (const packageDir of installedPackages) {
      const fullPackageName = `${config.scope}/${packageDir}`;

      // Check if this package matches the target platform
      if (!matchesTargetPlatform(fullPackageName, targetPlatform.patterns)) {
        const packagePath = path.join(scopeDir, packageDir);
        console.log(`  Removing ${fullPackageName} (does not match ${targetPlatform.target})`);
        try {
          fs.rmSync(packagePath, { recursive: true, force: true });
        } catch (error) {
          console.warn(`  ⚠️  Failed to remove ${fullPackageName}: ${error.message}`);
        }
      }
    }
  }
}

async function main() {
  // Clean up platform-specific packages that don't match the target
  // This ensures each package operation only includes dependencies for the target platform
  cleanupOtherPlatformPackages(targetPlatform);

  // Ensure necessary directories exist for all configured package scopes
  const scopeDirs = {};

  for (const config of PLATFORM_PACKAGE_CONFIGS) {
    const dir = path.join(__dirname, '..', 'node_modules', config.scope);
    scopeDirs[config.scope] = dir;

    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
  }

  for (const { name, version, config } of platformDeps) {
    // Get the base directory for this package's scope
    const baseDir = scopeDirs[config.scope];
    const packageDir = path.join(baseDir, name.split('/')[1]);

    // Skip if already installed
    if (fs.existsSync(packageDir)) {
      console.log(`  ✓ ${name} already installed`);
      // Still remove platform restrictions
      await removePlatformRestrictions(packageDir);
      continue;
    }

    // Create package directory
    fs.mkdirSync(packageDir, { recursive: true });

    try {
      await downloadAndExtract(name, version, packageDir);
      await removePlatformRestrictions(packageDir);
      console.log(`  ✓ ${name} installed`);
    } catch (error) {
      console.error(`  ✗ Failed to install ${name}: ${error.message}`);
      // Clean up partial installation
      if (fs.existsSync(packageDir)) {
        fs.rmSync(packageDir, { recursive: true, force: true });
      }
    }
  }

  console.log('\nVerifying installations:');

  // Verify each configured package type for target platform
  let allInstalled = true;

  for (const config of PLATFORM_PACKAGE_CONFIGS) {
    const scopeDir = path.join(__dirname, '..', 'node_modules', config.scope);

    if (!fs.existsSync(scopeDir)) {
      console.warn(`\n⚠️  ${config.description}: Directory ${config.scope} not found`);
      allInstalled = false;
      continue;
    }

    // Filter to only packages matching target platform
    const installed = fs.readdirSync(scopeDir)
      .filter(dir => {
        if (!dir.startsWith(config.prefix)) return false;
        const fullPackageName = `${config.scope}/${dir}`;
        return matchesTargetPlatform(fullPackageName, targetPlatform.patterns);
      });

    console.log(`\n${config.description}: Found ${installed.length} platform-specific module(s) for ${targetPlatform.target}:`);
    installed.forEach(dir => console.log(`  ✓ ${dir}`));

    if (installed.length < config.expectedCount) {
      console.warn(`\n⚠️  Warning: Expected ${config.expectedCount} ${config.description} platform module(s) for ${targetPlatform.target} but found ${installed.length}`);
      allInstalled = false;
    }
  }

  if (allInstalled) {
    console.log(`\n✓ All platform-specific dependencies for ${targetPlatform.target} installed successfully!`);
  }
}

main().catch(error => {
  console.error('\n✗ Failed to install platform-specific dependencies');
  console.error(error.message);
  process.exit(1);
});
