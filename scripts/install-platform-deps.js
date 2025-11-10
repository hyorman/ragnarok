#!/usr/bin/env node

/**
 * Install platform-specific dependencies for packaging VS Code extension
 *
 * This script ensures all platform-specific native modules are installed
 * before packaging, so the extension works on all platforms regardless of
 * which platform it was built on.
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
    expectedCount: 8,
    sources: ['dependencies', 'optionalDependencies']
  },
  {
    scope: '@img',
    prefix: 'sharp-',
    description: 'Sharp',
    expectedCount: 6,
    sources: ['dependencies', 'optionalDependencies']
  }
];

// Read package.json
const packageJsonPath = path.join(__dirname, '..', 'package.json');
const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));

console.log('Installing platform-specific dependencies for cross-platform packaging...\n');

// Collect platform-specific packages based on configuration
const platformDeps = [];

for (const config of PLATFORM_PACKAGE_CONFIGS) {
  for (const source of config.sources) {
    const deps = packageJson[source] || {};
    const filtered = Object.entries(deps)
      .filter(([name]) => name.startsWith(`${config.scope}/${config.prefix}`))
      .map(([name, version]) => ({
        name,
        version: version.replace(/^[\^~]/, ''),
        config
      }));
    platformDeps.push(...filtered);
  }
}

if (platformDeps.length === 0) {
  console.log('No platform-specific dependencies found.');
  process.exit(0);
}

console.log('Target platform-specific binaries:');
platformDeps.forEach(({name, version}) => console.log(`  - ${name}@${version}`));
console.log();async function downloadAndExtract(packageName, version, targetDir) {
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

async function main() {
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

  // Verify each configured package type
  let allInstalled = true;

  for (const config of PLATFORM_PACKAGE_CONFIGS) {
    const scopeDir = path.join(__dirname, '..', 'node_modules', config.scope);

    if (!fs.existsSync(scopeDir)) {
      console.warn(`\n⚠️  ${config.description}: Directory ${config.scope} not found`);
      allInstalled = false;
      continue;
    }

    const installed = fs.readdirSync(scopeDir).filter(dir => dir.startsWith(config.prefix));
    console.log(`\n${config.description}: Found ${installed.length} platform-specific modules:`);
    installed.forEach(dir => console.log(`  ✓ ${dir}`));

    if (installed.length < config.expectedCount) {
      console.warn(`\n⚠️  Warning: Expected ${config.expectedCount} ${config.description} platform modules but found ${installed.length}`);
      console.warn('Some platforms may not work correctly.');
      allInstalled = false;
    }
  }

  if (allInstalled) {
    console.log('\n✓ All platform-specific dependencies installed successfully!');
  }
}

main().catch(error => {
  console.error('\n✗ Failed to install platform-specific dependencies');
  console.error(error.message);
  process.exit(1);
});
