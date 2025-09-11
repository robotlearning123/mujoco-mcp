# OpenAI ChatGPT MCP Integration

Since OpenAI ChatGPT doesn't natively support MCP, we provide a bridge that connects ChatGPT's function calling capabilities with MCP servers.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Get OpenAI API Key**:
   - Visit [OpenAI API Keys](https://platform.openai.com/api-keys)
   - Create a new API key
   - Add billing information to your OpenAI account

3. **Configure Environment**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   export MUJOCO_GL="osmesa"  # For headless rendering
   ```

4. **Install MuJoCo MCP**:
   ```bash
   # From the root directory
   pip install -e .
   ```

## Usage

### Command Line Bridge
```bash
python mcp-bridge.py
```

This starts an interactive session where you can:
- Chat with GPT-4 enhanced with MuJoCo MCP functions
- Use `/servers` to list available MCP servers
- Use `/functions` to list available functions
- Use `/quit` to exit

### Example Conversation
```
You: Create a pendulum simulation and run it for 100 steps