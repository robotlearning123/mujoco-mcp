# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Which versions are eligible for receiving such patches depends on the CVSS v3.0 Rating:

| Version | Supported          |
| ------- | ------------------ |
| 0.8.x   | :white_check_mark: |
| < 0.8   | :x:                |

## Reporting a Vulnerability

We take the security of MuJoCo MCP seriously. If you believe you have found a security vulnerability, please report it to us as described below.

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to security@mujoco-mcp.org (or create a private security advisory on GitHub).

You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

Please include the requested information listed below (as much as you can provide) to help us better understand the nature and scope of the possible issue:

* Type of issue (e.g. buffer overflow, SQL injection, cross-site scripting, etc.)
* Full paths of source file(s) related to the manifestation of the issue
* The location of the affected source code (tag/branch/commit or direct URL)
* Any special configuration required to reproduce the issue
* Step-by-step instructions to reproduce the issue
* Proof-of-concept or exploit code (if possible)
* Impact of the issue, including how an attacker might exploit the issue

This information will help us triage your report more quickly.

## Preferred Languages

We prefer all communications to be in English.

## Security Update Policy

When we receive a security bug report, we will:

1. Confirm the problem and determine the affected versions
2. Audit code to find any potential similar problems
3. Prepare fixes for all supported releases
4. Release new security patch versions as soon as possible

## Security Best Practices

When using MuJoCo MCP, we recommend the following security practices:

### 1. Input Validation
- Always validate and sanitize XML input before loading MuJoCo models
- Verify model files come from trusted sources
- Use the built-in validation functions before loading models

### 2. Network Security
- When using the MCP server, bind to localhost (127.0.0.1) by default
- Use authentication and encryption when exposing the server over network
- Keep the server behind a firewall in production environments

### 3. Dependency Management
- Keep all dependencies up to date
- Regularly run `pip install --upgrade mujoco-mcp`
- Monitor security advisories for MuJoCo and other dependencies

### 4. Resource Limits
- Set appropriate timeouts for simulations
- Limit the complexity of models that can be loaded
- Monitor memory usage in long-running simulations

### 5. Code Execution
- Be cautious when loading models from untrusted sources
- Models can contain custom XML that may affect simulation behavior
- Review model files before loading in production environments

## Known Security Considerations

### XML External Entity (XXE) Attacks
MuJoCo XML parsing may be vulnerable to XXE attacks if external entities are enabled. We disable external entity resolution by default. Do not enable external entities when parsing untrusted XML.

### Code Injection via Model Files
MuJoCo model files can contain plugin references and custom elements. Only load models from trusted sources in production environments.

### Denial of Service (DoS)
Very large or complex models can consume significant CPU and memory. Implement resource limits when accepting models from untrusted sources.

## Security Hall of Fame

We would like to thank the following individuals for responsibly disclosing security issues:

* (No reports yet - be the first!)

## Disclosure Policy

When a security vulnerability is found:

1. We will work with the reporter to understand and verify the issue
2. We will develop and test a fix
3. We will release the fix and publish a security advisory
4. We will credit the reporter (unless they wish to remain anonymous)

We ask that you:
- Give us reasonable time to fix the issue before public disclosure
- Make a good faith effort to avoid privacy violations and service disruption
- Do not access or modify data that doesn't belong to you

Thank you for helping keep MuJoCo MCP and our users safe!
