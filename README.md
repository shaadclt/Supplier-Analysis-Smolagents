# Ice Cream Supplier Analysis using Smolagents🍦

This project demonstrates how to use smolagents to build a simple yet powerful agent that helps calculate transportation costs and tariffs for an ice cream supplier business.

## Overview

This project provides tools for analyzing ice cream supplier data, calculating transportation costs, tariffs, and total procurement expenses. It leverages the power of `smolagents` to create intelligent agents that can perform complex calculations and answer natural language queries about supplier data.

## About smolagents

This project showcases [smolagents](https://github.com/huggingface/smolagents), a powerful library that enables running sophisticated agents in just a few lines of code:

- ✨ **Simplicity**: The entire agent logic fits in ~1,000 lines of code with minimal abstractions
- 🧑‍💻 **First-class Code Agent support**: Unlike tools that use agents to write code, smolagents' CodeAgent writes its actions directly in code with secure execution in sandboxed environments (E2B or Docker)
- 🤗 **Hub integrations**: Share and pull tools or agents to/from the Hugging Face Hub
- 🌐 **Model-agnostic**: Works with any LLM (local transformers, ollama, Hub providers, OpenAI, Anthropic, and more via LiteLLM)
- 👁️ **Modality-agnostic**: Supports text, vision, video, and audio inputs
- 🛠️ **Tool-agnostic**: Compatible with tools from LangChain, MCP, or even Hub Spaces

## Features

- **Supplier Data Management**: Store and analyze supplier information including location, distance, pricing, and country of origin
- **Cost Calculation Tools**: 
  - Transportation cost calculation based on distance and volume
  - Tariff calculation for international imports
  - Total cost analysis including fixed fees
- **AI-Powered Analysis**: Uses smolagents with the Qwen2.5-72B-Instruct model for natural language interaction with your data

## Installation

```bash
# Clone the repository
git clone https://github.com/shaadclt/Supplier-Analysis-Smolagents.git
cd Supplier-Analysis-Smolagents
```

## Usage

### Basic Setup

```python
import pandas as pd
import numpy as np
from smolagents import tool, HfApiModel, CodeAgent

# Set up your HuggingFace API token
import os
os.environ['HF_API_KEY'] = "your_huggingface_token"

# Load your supplier data
suppliers_data = {
    "name": ["Montreal Ice Cream Co", "Brain Freeze Brothers", "Toronto Gelato Ltd"],
    "location": ["Montreal, QC", "Burlington, VT", "Toronto, ON"],
    "distance_km": [120, 85, 400],
    "canadian": [True, False, True],
    "price_per_liter": [1.95, 1.91, 1.82],
    "tasting_fee": [0, 12.50, 30.14]
}
suppliers_df = pd.DataFrame(suppliers_data)
```

### Creating Analysis Tools with smolagents

The power of smolagents lies in its simple yet powerful tool definition system:

```python
@tool
def calculate_transport_cost(distance_km: float, order_volume: float) -> float:
    """Calculate transportation cost based on distance and order size."""
    trucks_needed = np.ceil(order_volume / 300)
    cost_per_km = 1.20
    return distance_km * cost_per_km * trucks_needed

@tool
def calculate_tariff(base_cost: float, is_canadian: bool) -> float:
    """Calculates tariff for Canadian imports."""
    if is_canadian:
        return base_cost * np.pi / 50
    return 0
```

### Using the smolagents CodeAgent

```python
# Initialize a model using smolagents' model-agnostic approach
model = HfApiModel(
    "Qwen/Qwen2.5-72B-Instruct",
    provider="together",
    max_tokens=4096,
    temperature=0.1
)

# Create a CodeAgent that can use your custom tools
agent = CodeAgent(
    model=model,
    tools=[calculate_transport_cost, calculate_tariff],
    max_steps=10,
    additional_authorized_imports=["pandas", "numpy"],
    verbosity_level=2
)

# Ask questions in natural language and get programmatic responses
response = agent.run("What is the transportation cost for 50 liters of ice cream over 10 kilometers?")
print(response)
```

## Advanced smolagents Features

This project demonstrates several key features of smolagents:

1. **Tool Composition**: Combine multiple tools to solve complex supply chain problems
2. **Secure Code Execution**: Run agent-generated code in a sandboxed environment
3. **Natural Language Interface**: Convert user questions into programmatic operations
4. **Model Flexibility**: Easily swap between different LLMs for your specific needs

## Requirements

- Python 3.8+
- pandas
- numpy
- smolagents
- Hugging Face API access

## Google Colab Support

This project includes a Google Colab notebook for easy exploration without local setup. The notebook demonstrates how to leverage smolagents to:

1. Load and analyze supplier data
2. Calculate transportation costs and tariffs
3. Use the CodeAgent to answer natural language questions
4. Visualize cost comparisons between suppliers

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
