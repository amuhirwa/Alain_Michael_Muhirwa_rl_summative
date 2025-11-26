"""
Humanoid Model Animation Stripper
==================================

The humanoid model contains animation data that causes issues with panda3d-gltf.
This script creates a static version by removing animations.

Usage:
    python fix_humanoid_model.py
"""

import json
import os
import shutil
from pathlib import Path

def strip_animations_from_gltf():
    """Remove animation data from humanoid GLTF file"""
    
    gltf_path = Path("3d_models/humanoid/scene.gltf")
    backup_path = Path("3d_models/humanoid/scene.gltf.backup")
    
    if not gltf_path.exists():
        print(f"❌ {gltf_path} not found!")
        return False
    
    print("=" * 60)
    print("Humanoid Model Animation Stripper")
    print("=" * 60)
    
    # Backup original
    if not backup_path.exists():
        print(f"\n📦 Creating backup: {backup_path}")
        shutil.copy(gltf_path, backup_path)
    
    # Load GLTF JSON
    print(f"\n📖 Reading {gltf_path}...")
    with open(gltf_path, 'r', encoding='utf-8') as f:
        gltf_data = json.load(f)
    
    # Check for animations
    if 'animations' not in gltf_data or len(gltf_data['animations']) == 0:
        print("   ✓ No animations found, model is already static")
        return True
    
    anim_count = len(gltf_data['animations'])
    print(f"   📊 Found {anim_count} animation(s)")
    
    # Remove animations
    print(f"\n✂️  Removing animations...")
    del gltf_data['animations']
    
    # Save modified GLTF
    print(f"\n💾 Saving static version to {gltf_path}...")
    with open(gltf_path, 'w', encoding='utf-8') as f:
        json.dump(gltf_data, f, indent=2)
    
    print(f"   ✓ Removed {anim_count} animation(s)")
    print(f"\n   Original saved as: {backup_path}")
    
    print("\n" + "=" * 60)
    print("Success! Now run: python convert_gltf_models.py")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    print("\nFixing humanoid model...\n")
    
    success = strip_animations_from_gltf()
    
    if success:
        print("\n✅ Humanoid model is now ready for conversion!")
        print("\nNext steps:")
        print("  1. Run: python convert_gltf_models.py")
        print("  2. Run: python main.py --model models/a2c/config_2_high_lr/best_model.zip")
    else:
        print("\n❌ Failed to fix humanoid model")
        print("\nAlternative: Use door model only (enhanced_rendering.py will fallback to")
        print("             procedural geometry for agents, which still looks great!)")
