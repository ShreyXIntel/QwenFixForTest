# 🎮 Automated Game Benchmarking System

An AI-powered system that autonomously navigates through game menus, locates and runs built-in benchmarks, captures results, and cleanly exits - all without human intervention.

![Game Benchmarking System](https://i.imgur.com/example.png)

## 🚀 Overview

The Automated Game Benchmarking System uses computer vision and AI to intelligently navigate through game interfaces, eliminating the tedium of manual benchmark testing. It leverages the Qwen 2.5 VL vision-language model to interpret UI elements and make navigation decisions with high accuracy.

### Key Features

- **Fully Autonomous Operation**: Complete benchmark workflow without human input
- **Game-Specific Intelligence**: Pre-configured navigation flows for popular games
- **Vision-Language AI**: Qwen 2.5 VL model for advanced UI understanding
- **Modular Architecture**: Separates concerns for easy maintenance and extension
- **Detailed Result Tracking**: Comprehensive logging, screenshots, and performance metrics
- **Visual Debugging**: Annotated screenshots showing detection and decision-making
- **Graceful Process Handling**: Proper launch and exit sequences for games

## 🧠 How It Works

1. **UI Analysis**: Screenshots the game interface and uses Qwen 2.5 VL to identify UI elements
2. **Navigation Planning**: Determines the optimal path to find and run benchmark options
3. **Interaction**: Uses Win32 API to simulate mouse and keyboard inputs with precise timing
4. **Result Detection**: Automatically detects when benchmarks complete and captures result screens
5. **Clean Exit**: Navigates back to the main menu and properly exits the game

## 🔧 Technical Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  GameBenchmarker│────│    UIAnalyzer   │────│   Qwen 2.5 VL   │
└────────┬────────┘    └─────────────────┘    └─────────────────┘
         │
┌────────┴────────┐    ┌─────────────────┐    ┌─────────────────┐
│  FlowManager    │────│  ResultDetector │    │ ScreenshotMgr   │
└────────┬────────┘    └─────────────────┘    └─────────────────┘
         │
┌────────┴────────┐
│ InputController │
└─────────────────┘
```

## 📋 Requirements

- Windows 10/11 (required for Win32 API)
- Python 3.8+
- NVIDIA GPU with at least 8GB VRAM (for Qwen 2.5 VL model)
- CUDA 11.7+ and cuDNN
- Games with built-in benchmarking functionality
- 16GB+ system RAM recommended

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ShreyXIntel/QwenFixForTest.git
   cd game-benchmarker
   ```

2. **You need not set up a virtual environment**
   We are facing some issues while installing dependencies and packages to a venv. Instead when installed in global env. the project works.

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Qwen 2.5 VL model:**
   ```bash
   # This will be downloaded automatically on first run,
   # or can be downloaded manually from Hugging Face
   ```

5. **Verify installation:**
   ```bash
   python launcher.py --verify
   ```

## 🎮 Supported Games

The system includes pre-configured navigation flows for:

- Black Myth: Wukong Benchmark
- Far Cry 6
- Assassin's Creed series
- Counter-Strike 2
- Cyberpunk 2077

New games can be added by extending the `flow.json` configuration.

## 🖥️ Usage

### Basic Usage

With a game already running:

```bash
python launcher.py
```

### Launch and Benchmark a Game

```bash
python launcher.py --game "C:\Path\to\game.exe"
```

### Full Options

```bash
python launcher.py --game "C:\Path\to\game.exe" --config custom_config.py --flow custom_flow.json --timeout 300 --screenshot-interval 1.5
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to configuration file | `config.py` |
| `--game` | Path to game executable | None (assumes game is running) |
| `--flow` | Path to flow configuration file | `flow.json` |
| `--timeout` | Override benchmark timeout (seconds) | From config |
| `--screenshot-interval` | Override screenshot interval (seconds) | From config |
| `--verify` | Run verification tests before starting | False |

## 📂 Output Structure

Each benchmark run creates a timestamped directory under `benchmark_runs/`:

```
benchmark_runs/
└── run_20250609_153045/
    ├── Raw Screenshots/       # All screenshots captured during navigation
    ├── Analyzed Screenshots/  # Annotated images showing detection results
    ├── Logs/                  # Detailed logs and JSON summary
    └── Benchmark Results/     # Screenshots of benchmark result screens
```

## ⚙️ Configuration

### `config.py`

Central configuration file with sections for:

- **MODEL_CONFIG**: Qwen 2.5 VL model settings
- **BENCHMARK_CONFIG**: Timeouts, intervals, and navigation parameters
- **UI_PROMPT**: Instructions for the vision model
- **DEBUG_CONFIG**: Logging and visualization options

### `flow.json`

Game-specific navigation configurations:

- Detection patterns for identifying games
- Menu navigation sequences
- Priority UI elements
- Benchmark indicator keywords
- Exit strategies

Example for adding a new game:

```json
{
  "game_name": "Your Game",
  "detection_hints": [
    {
      "context": "Main Menu",
      "priority_elements": ["OPTIONS", "SETTINGS", "BENCHMARK"],
      "navigation_path": [
        {
          "menu": "Main Menu",
          "click": "OPTIONS"
        },
        {
          "menu": "Options Menu",
          "click": "GRAPHICS"
        },
        {
          "menu": "Graphics Options",
          "click": "BENCHMARK"
        }
      ],
      "benchmark_indicators": ["FPS", "BENCHMARK RESULTS"],
      "back_to_main_menu": [
        {
          "action": "PRESS_KEY",
          "key": "escape"
        }
      ]
    }
  ]
}
```

## 🛑 Limitations

- Works only on Windows due to Win32 API dependency
- Requires games with built-in benchmark functionality
- Performance depends on GPU capabilities for AI model inference
- May require game-specific tuning for optimal results
- Does not support games running in exclusive fullscreen mode

## 🔍 Troubleshooting

- **Navigation loops**: Increase the `confidence_threshold` in config.py
- **Model fails to load**: Check CUDA installation and GPU VRAM availability
- **Click positions incorrect**: Ensure game is running at the same resolution as when creating flow.json
- **Benchmark not detected**: Add more benchmark indicators to flow.json

## 📊 Performance Considerations

- Model inference speed depends on GPU capability
- Screenshot interval affects navigation speed (lower = faster but more resource intensive)
- Adjust `navigation_delay` for games with slower menu transitions

## 🧩 Extending the System

- Add new games by extending `flow.json`
- Improve UI detection by updating the UI prompt in `config.py`
- Implement new navigation strategies in `flow_manager.py`
- Add result parsing functionality to extract benchmark metrics

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Related Projects

- [Benchmark Result Analyzer](https://github.com/example/analyzer)
- [Game Performance Database](https://github.com/example/perfdb)

## 🙏 Acknowledgments

- Qwen team for the vision-language model
- Game developers who include benchmarking tools
- Open source computer vision and AI communities