"""
Complete Workflow - Train, Evaluate, and Generate Report Materials
===================================================================

This script automates the entire process:
1. Train PPO and DQN
2. Generate all plots
3. Test generalization
4. Create summary report

Run this to get everything needed for 50/50!
"""

import subprocess
import sys
import os
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def check_prerequisites():
    """Check if all required files exist"""
    required_files = [
        "environment/enhanced_env_fast.py",
        "training/quick_train.py",
        "evaluation/generate_report_plots.py",
        "demo.py"
    ]
    
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
    
    if missing:
        print("❌ Missing required files:")
        for file in missing:
            print(f"  - {file}")
        return False
    
    print("✅ All required files present")
    return True

def main():
    print_header("COMPLETE RL SUMMATIVE WORKFLOW")
    print("This will:")
    print("  1. Train PPO and DQN models (~30-60 minutes)")
    print("  2. Generate all report plots (~5-10 minutes)")
    print("  3. Test generalization across scenarios")
    print("  4. Create summary statistics")
    print("\nTotal time: ~40-70 minutes")
    print("\nPress Ctrl+C to cancel, or Enter to continue...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        return
    
    # Check prerequisites
    print_header("STEP 0: CHECKING PREREQUISITES")
    if not check_prerequisites():
        print("\n❌ Prerequisites check failed. Please ensure all files are present.")
        return
    
    # Step 1: Train models
    print_header("STEP 1: TRAINING MODELS")
    print("Training PPO and DQN with 50,000 timesteps each...")
    print("This may take 30-60 minutes depending on your hardware.\n")
    
    success = run_command(
        "uv run python training/quick_train.py --timesteps 50000 --algorithm both",
        "Model training"
    )
    
    if not success:
        print("\n❌ Training failed. Check error messages above.")
        print("You can try reducing timesteps:")
        print("  uv run python training/quick_train.py --timesteps 30000 --algorithm both")
        return
    
    # Check if models were created
    ppo_model = Path("models/quick_ppo/ppo_final.zip")
    dqn_model = Path("models/quick_dqn/dqn_final.zip")
    
    if not ppo_model.exists() or not dqn_model.exists():
        print("\n❌ Models were not created successfully.")
        print("Expected files:")
        print(f"  - {ppo_model} {'✅' if ppo_model.exists() else '❌'}")
        print(f"  - {dqn_model} {'✅' if dqn_model.exists() else '❌'}")
        return
    
    print(f"\n✅ Models saved:")
    print(f"  - {ppo_model}")
    print(f"  - {dqn_model}")
    
    # Step 2: Generate plots
    print_header("STEP 2: GENERATING REPORT PLOTS")
    
    success = run_command(
        "uv run python evaluation/generate_report_plots.py",
        "Plot generation"
    )
    
    if not success:
        print("\n⚠️ Plot generation had issues, but may have created some plots.")
        print("Check the results/ folder.")
    
    # Check generated files
    expected_plots = [
        "results/1_cumulative_rewards.png",
        "results/2_training_metrics.png",
        "results/3_convergence.png",
        "results/4_generalization.png",
        "results/5_performance_summary.png"
    ]
    
    print("\n📊 Generated plots:")
    for plot in expected_plots:
        exists = Path(plot).exists()
        print(f"  {'✅' if exists else '❌'} {plot}")
    
    # Step 3: Summary
    print_header("WORKFLOW COMPLETE!")
    
    print("✅ WHAT YOU HAVE NOW:\n")
    
    if ppo_model.exists():
        print("1. ✅ Trained PPO model")
        print("   Location: models/quick_ppo/ppo_final.zip")
        print("   Metrics: models/quick_ppo/results.json\n")
    
    if dqn_model.exists():
        print("2. ✅ Trained DQN model")
        print("   Location: models/quick_dqn/dqn_final.zip")
        print("   Metrics: models/quick_dqn/results.json\n")
    
    plots_exist = sum(Path(p).exists() for p in expected_plots)
    if plots_exist > 0:
        print(f"3. ✅ Report plots ({plots_exist}/5)")
        print("   Location: results/ folder\n")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("\n1. 🎥 RECORD DEMO VIDEO (3 minutes max, camera on, full screen):")
    print("   Command:")
    print("   uv run python demo.py --difficulty medium --pattern rush --model models/quick_ppo/ppo_final.zip")
    print("\n2. 📝 FILL OUT REPORT:")
    print("   - Use plots from results/ folder")
    print("   - Copy metrics from models/quick_*/results.json")
    print("   - Follow template structure")
    print("\n3. ✅ FINAL CHECKLIST:")
    print("   [ ] Video recorded and uploaded (YouTube/Drive)")
    print("   [ ] Report completed with all sections")
    print("   [ ] Hyperparameter tables filled (10+ rows)")
    print("   [ ] All 5 plots included with descriptions")
    print("   [ ] GitHub repository updated")
    print("   [ ] Video link added to report")
    print("\n4. 🚀 SUBMIT!")
    print("\n" + "="*70)
    print("\n💡 TIP: Open results/ folder to view all plots:")
    print("   explorer results")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
