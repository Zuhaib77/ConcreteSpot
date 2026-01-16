import shutil
import uuid
from pathlib import Path
from datetime import datetime


class Storage:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.images_dir = data_dir / "images"
        self.reports_dir = data_dir / "reports"
        
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def save_image(self, source_path: Path) -> Path:
        date_folder = self.images_dir / datetime.now().strftime("%Y-%m-%d")
        date_folder.mkdir(exist_ok=True)
        
        unique_id = uuid.uuid4().hex[:8]
        ext = source_path.suffix
        new_name = f"{unique_id}{ext}"
        dest_path = date_folder / new_name
        
        shutil.copy2(source_path, dest_path)
        return dest_path
    
    def get_report_path(self, analysis_id: int) -> Path:
        date_folder = self.reports_dir / datetime.now().strftime("%Y-%m-%d")
        date_folder.mkdir(exist_ok=True)
        return date_folder / f"report_{analysis_id}.pdf"
    
    def get_all_images(self) -> list[Path]:
        images = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            images.extend(self.images_dir.rglob(ext))
        return sorted(images, key=lambda x: x.stat().st_mtime, reverse=True)
