import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class Visualizer:
    def __init__(self, save_dir="graphs"):
        self.save_dir = save_dir

        Path(save_dir).mkdir(parents=True, exist_ok=True)

        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 100

    def _save_and_show(self, save_path: str, show: bool = True):
        full_path = f"{self.save_dir}/{save_path}.png"
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        logger.info(f'Graph saved in {full_path}')
        if show:
            plt.show()
        plt.close()

    def countplot(self, data: DataFrame, x: str, hue: str, title: str, 
                  xlabel: str, ylabel:str, save_path:str, 
                  palette: str | list[str], figsize: tuple[int, int] = (10, 6)):
        plt.figure(figsize=figsize)

        order = data[x].value_counts().index if data[x].dtype == "object" else None

        ax = sns.countplot(
            data=data, 
            x=x, 
            hue=hue, 
            palette=palette,
            order=order
        )
        for p in ax.patches:
            ax.annotate(
                f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center',
                va='center',
                xytext=(0, 10),
                textcoords='offset points'
            )
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.legend(title=hue)
        plt.tight_layout()
        
        self._save_and_show(save_path)
    
    def barplot(self, data: DataFrame, x: str, y: str, title: str, 
                xlabel: str, ylabel: str, save_path: str, 
                palette: str | list[str], figsize: tuple[int, int] = (10, 6)):
        plt.figure(figsize=figsize)

        ax = sns.barplot(
            data=data,
            x=x, 
            y=y, 
            palette=palette
        )

        for i, v in enumerate(data[x]):
            ax.text(
                v + 0.01, i , f'{v:.3f}',
                va='center', fontsize=9
            )

        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.tight_layout()

        self._save_and_show(save_path)
    
    def heatmap(self, data: DataFrame, annot: bool = True, 
                cmap: str = 'coolwarm', fmt: str = '.2f', 
                linewidths: float = 0.5, title: str = '', 
                save_path: str = 'heatmap', figsize: tuple[int, int] = (10, 8)):
        plt.figure(figsize=figsize)

        sns.heatmap(
            data=data, 
            annot=annot,
            cmap=cmap, 
            fmt=fmt, 
            linewidths=linewidths,
            square=True,
            cbar_kws={"shrink": 0.8}
        )

        if title:
            plt.title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        self._save_and_show(save_path)

    def plot_confusion_matrix(self, cm, title: str = 'Confusion Matrix',
                              save_path: str = 'confusion_matrix'):
        plt.figure(figsize=(8,6))

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=False,
            square=True
        )

        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.tight_layout()

        self._save_and_show(save_path)