#!/usr/bin/env python3
"""
Engine version manager - saves and compares different engine configurations.
"""
import os
import sys
import json
import shutil
import subprocess
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VERSIONS_DIR = os.path.join(os.path.dirname(__file__), 'engine_versions')
os.makedirs(VERSIONS_DIR, exist_ok=True)

def save_current_version(name=None, description=""):
    """Save current engine state as a version"""
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"v_{timestamp}"
    
    version_dir = os.path.join(VERSIONS_DIR, name)
    os.makedirs(version_dir, exist_ok=True)
    
    # Copy key engine files
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files_to_save = [
        'chess_engine/tuned_weights.py',
        'chess_engine/search.py',
        'chess_engine/eval_main.py',
        'chess_engine/evaluation/features_tactics.py',
        'chess_engine/evaluation/features_king_safety.py',
        'chess_engine/evaluation/features_pawn.py',
        'chess_engine/evaluation/features_piece_activity.py',
        'chess_engine/evaluation/features_material.py',
    ]
    
    for rel_path in files_to_save:
        src = os.path.join(base_dir, rel_path)
        if os.path.exists(src):
            dst_dir = os.path.join(version_dir, os.path.dirname(rel_path))
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src, os.path.join(version_dir, rel_path))
    
    # Save metadata
    meta = {
        'name': name,
        'timestamp': datetime.now().isoformat(),
        'description': description
    }
    with open(os.path.join(version_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"Saved version: {name}")
    return name

def list_versions():
    """List all saved versions"""
    versions = []
    for name in os.listdir(VERSIONS_DIR):
        meta_path = os.path.join(VERSIONS_DIR, name, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            versions.append(meta)
    return sorted(versions, key=lambda x: x['timestamp'], reverse=True)

def restore_version(name):
    """Restore a saved version"""
    version_dir = os.path.join(VERSIONS_DIR, name)
    if not os.path.exists(version_dir):
        print(f"Version not found: {name}")
        return False
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Restore files
    for root, dirs, files in os.walk(version_dir):
        for file in files:
            if file.endswith('.py'):
                src = os.path.join(root, file)
                rel_path = os.path.relpath(src, version_dir)
                dst = os.path.join(base_dir, rel_path)
                shutil.copy2(src, dst)
                print(f"  Restored: {rel_path}")
    
    print(f"Restored version: {name}")
    return True

def benchmark_version(name, num_games=8, depth=3):
    """Benchmark a specific version"""
    print(f"Benchmarking version: {name}")
    
    # Temporarily restore version
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Run benchmark
    result = subprocess.run([
        sys.executable, 
        os.path.join(os.path.dirname(__file__), 'quick_test.py'),
        '--depth', str(depth),
        '--games', str(num_games)
    ], capture_output=True, text=True, cwd=base_dir)
    
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    return result.returncode == 0

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['save', 'list', 'restore', 'benchmark'])
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--desc', type=str, default="")
    args = parser.parse_args()
    
    if args.action == 'save':
        save_current_version(args.name, args.desc)
    elif args.action == 'list':
        versions = list_versions()
        print("Saved versions:")
        for v in versions:
            print(f"  {v['name']}: {v.get('description', '')[:50]} ({v['timestamp'][:10]})")
    elif args.action == 'restore':
        if args.name:
            restore_version(args.name)
        else:
            print("Please specify --name")
    elif args.action == 'benchmark':
        if args.name:
            benchmark_version(args.name)
        else:
            print("Please specify --name")
