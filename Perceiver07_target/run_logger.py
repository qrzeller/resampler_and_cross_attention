# -*- coding: utf-8 -*-
"""
Run logging utilities for automatic documentation of training runs.

Creates a timestamped folder with:
- Markdown report (README.md) containing command, comments, config, and results
- Training history plot
- Reconstruction plots
- Checkpoint (optional)
"""

from __future__ import annotations

import datetime
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RunLogger:
    """
    Logger for documenting training runs.
    
    Creates a structured run folder with all outputs and a markdown report.
    """
    
    output_base: Path = field(default_factory=lambda: Path("runs"))
    run_name: Optional[str] = None
    comment: Optional[str] = None
    command: Optional[str] = None
    
    # Paths set after initialization
    run_dir: Path = field(init=False)
    recon_fig_path: Path = field(init=False)
    history_fig_path: Path = field(init=False)
    checkpoint_path: Path = field(init=False)
    report_path: Path = field(init=False)
    
    # Collected data
    config: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, float]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Create the run directory and set up paths."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.run_name:
            # Sanitize run name for filesystem
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.run_name)
            folder_name = f"{timestamp}_{safe_name}"
        else:
            folder_name = timestamp
        
        self.run_dir = self.output_base / folder_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up output paths
        self.recon_fig_path = self.run_dir / "reconstruction.png"
        self.history_fig_path = self.run_dir / "training_history.png"
        self.checkpoint_path = self.run_dir / "checkpoint.pt"
        self.report_path = self.run_dir / "README.md"
        
        # Capture the command that was run
        if self.command is None:
            self.command = " ".join(sys.argv)
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Store configuration for the report."""
        self.config = dict(config)
    
    def set_history(self, history: List[Dict[str, float]]) -> None:
        """Store training history for the report."""
        self.history = list(history)
    
    def add_metric(self, name: str, value: Any) -> None:
        """Add a metric to include in the report."""
        self.metrics[name] = value
    
    def generate_report(self) -> Path:
        """
        Generate the markdown report with all run information.
        
        Returns:
            Path to the generated report
        """
        lines = []
        
        # Header
        lines.append("# Training Run Report")
        lines.append("")
        lines.append(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Comment section (if provided)
        if self.comment:
            lines.append("## 💡 Notes / Goal")
            lines.append("")
            lines.append(self.comment)
            lines.append("")
        
        # Command
        lines.append("## 🚀 Command")
        lines.append("")
        lines.append("```bash")
        lines.append(self.command)
        lines.append("```")
        lines.append("")
        
        # Configuration
        if self.config:
            lines.append("## ⚙️ Configuration")
            lines.append("")
            lines.append("| Parameter | Value |")
            lines.append("|-----------|-------|")
            for key, value in sorted(self.config.items()):
                # Skip internal/path parameters
                if key in ("recon_fig", "history_fig", "save_checkpoint", "run_output"):
                    continue
                lines.append(f"| `{key}` | `{value}` |")
            lines.append("")
        
        # Training Results
        if self.history:
            lines.append("## 📊 Training Results")
            lines.append("")
            
            # Summary statistics
            final = self.history[-1]
            best_val = min(h.get("val_loss", float("inf")) for h in self.history)
            best_train = min(h.get("train_loss", float("inf")) for h in self.history)
            
            lines.append(f"- **Final Train Loss:** {final.get('train_loss', 'N/A'):.6f}")
            lines.append(f"- **Final Val Loss:** {final.get('val_loss', 'N/A'):.6f}")
            lines.append(f"- **Best Train Loss:** {best_train:.6f}")
            lines.append(f"- **Best Val Loss:** {best_val:.6f}")
            lines.append(f"- **Total Epochs:** {len(self.history)}")
            lines.append("")
            
            # Training history table
            lines.append("### Epoch-by-Epoch History")
            lines.append("")
            lines.append("| Epoch | Train Loss | Val Loss |")
            lines.append("|-------|------------|----------|")
            for h in self.history:
                epoch = h.get("epoch", "?")
                train_loss = h.get("train_loss", 0)
                val_loss = h.get("val_loss", 0)
                lines.append(f"| {epoch} | {train_loss:.6f} | {val_loss:.6f} |")
            lines.append("")
        
        # Additional metrics
        if self.metrics:
            lines.append("## 📈 Additional Metrics")
            lines.append("")
            for name, value in self.metrics.items():
                if isinstance(value, float):
                    lines.append(f"- **{name}:** {value:.6f}")
                else:
                    lines.append(f"- **{name}:** {value}")
            lines.append("")
        
        # Visualizations
        lines.append("## 📷 Visualizations")
        lines.append("")
        
        if self.history_fig_path.exists():
            lines.append("### Training History")
            lines.append("")
            lines.append(f"![Training History]({self.history_fig_path.name})")
            lines.append("")
        
        if self.recon_fig_path.exists():
            lines.append("### Reconstruction Examples")
            lines.append("")
            lines.append(f"![Reconstruction]({self.recon_fig_path.name})")
            lines.append("")
        
        # Checkpoint info
        if self.checkpoint_path.exists():
            lines.append("## 💾 Checkpoint")
            lines.append("")
            lines.append(f"Saved to: `{self.checkpoint_path.name}`")
            lines.append("")
        
        # Files in this run
        lines.append("## 📁 Files")
        lines.append("")
        for f in sorted(self.run_dir.iterdir()):
            size_kb = f.stat().st_size / 1024
            lines.append(f"- `{f.name}` ({size_kb:.1f} KB)")
        lines.append("")
        
        # Write report
        report_content = "\n".join(lines)
        self.report_path.write_text(report_content, encoding="utf-8")
        
        return self.report_path
    
    def get_paths(self) -> Dict[str, str]:
        """
        Get all output paths as strings for use in main.py.
        
        Returns:
            Dictionary with keys: recon_fig, history_fig, checkpoint, report, run_dir
        """
        return {
            "recon_fig": str(self.recon_fig_path),
            "history_fig": str(self.history_fig_path),
            "checkpoint": str(self.checkpoint_path),
            "report": str(self.report_path),
            "run_dir": str(self.run_dir),
        }


def create_run_logger(
    output_base: str = "runs",
    run_name: Optional[str] = None,
    comment: Optional[str] = None,
) -> RunLogger:
    """
    Factory function to create a RunLogger.
    
    Args:
        output_base: Base directory for all runs (default: "runs")
        run_name: Optional name for this run (will be appended to timestamp)
        comment: Optional comment/goal for this run
    
    Returns:
        Configured RunLogger instance
    """
    return RunLogger(
        output_base=Path(output_base),
        run_name=run_name,
        comment=comment,
    )
