"""
GLTF to Panda3D Converter
==========================

Converts GLTF models to Panda3D's native .bam format for faster loading.

Usage:
    python convert_gltf_models.py

Requirements:
    pip install panda3d-gltf

This script converts:
- 3d_models/humanoid/scene.gltf → scene.bam
- 3d_models/door/scene.gltf → scene.bam
"""

import os
import sys

try:
    from direct.showbase.ShowBase import ShowBase
    from panda3d.core import Filename
    import gltf
    GLTF_AVAILABLE = True
except ImportError:
    print("ERROR: panda3d-gltf not installed!")
    print("Install with: pip install panda3d-gltf")
    sys.exit(1)


class ModelConverter(ShowBase):
    """Simple converter application"""
    
    def __init__(self):
        super().__init__()
        self.convert_models()
        sys.exit(0)
    
    def convert_models(self):
        """Convert GLTF models to BAM format"""
        models_to_convert = [
            ("3d_models/humanoid/scene.gltf", "3d_models/humanoid/scene.bam"),
            ("3d_models/door/scene.gltf", "3d_models/door/scene.bam"),
        ]
        
        print("=" * 60)
        print("GLTF to Panda3D BAM Converter")
        print("=" * 60)
        
        for gltf_path, bam_path in models_to_convert:
            if not os.path.exists(gltf_path):
                print(f"\n❌ {gltf_path} not found, skipping...")
                continue
            
            print(f"\n📦 Converting {gltf_path}...")
            
            try:
                # Load GLTF model
                model = self.loader.loadModel(gltf_path)
                
                if model is None or model.isEmpty():
                    print(f"   ❌ Failed to load {gltf_path}")
                    continue
                
                # Save as BAM
                success = model.writeBamFile(bam_path)
                
                if success:
                    file_size = os.path.getsize(bam_path) / 1024  # KB
                    print(f"   ✓ Saved to {bam_path} ({file_size:.1f} KB)")
                else:
                    print(f"   ❌ Failed to save {bam_path}")
            
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        print("\n" + "=" * 60)
        print("Conversion complete!")
        print("=" * 60)
        print("\nThe enhanced_rendering.py will now load .bam files for faster performance.")


if __name__ == "__main__":
    print("\nStarting model conversion...")
    print("(This may take a few seconds)\n")
    
    converter = ModelConverter()
