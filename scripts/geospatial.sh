#!/bin/sh
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
LOG_FILE="$PROJECT_ROOT/logs/geospatial.log"

date > "$LOG_FILE"
echo "---------------------------------------------------" >> "$LOG_FILE"
python "$PROJECT_ROOT/scripts/01_geospatial_gradient_scan.py" >> "$LOG_FILE" 2>&1
echo "---------------------------------------------------" >> "$LOG_FILE"
python "$PROJECT_ROOT/scripts/02_geospatial_robust_validation.py" >> "$LOG_FILE" 2>&1
echo "---------------------------------------------------" >> "$LOG_FILE"
python "$PROJECT_ROOT/scripts/03_littoral_balanced_geometry.py" >> "$LOG_FILE" 2>&1
echo "---------------------------------------------------" >> "$LOG_FILE"
python "$PROJECT_ROOT/scripts/04_littoral_axis_hypothesis_tests.py" >> "$LOG_FILE" 2>&1
echo "---------------------------------------------------" >> "$LOG_FILE"
python "$PROJECT_ROOT/scripts/05_littoral_spectral_decomposition.py" >> "$LOG_FILE" 2>&1
echo "---------------------------------------------------" >> "$LOG_FILE"
python "$PROJECT_ROOT/scripts/06_littoral_quadrupole_validation.py" >> "$LOG_FILE" 2>&1
echo "---------------------------------------------------" >> "$LOG_FILE"
python "$PROJECT_ROOT/scripts/07_littoral_rotation_inverse.py" >> "$LOG_FILE" 2>&1
echo "---------------------------------------------------" >> "$LOG_FILE"
python "$PROJECT_ROOT/scripts/08_littoral_joint_rotation_quadrupole_inverse.py" >> "$LOG_FILE" 2>&1
echo "---------------------------------------------------" >> "$LOG_FILE"
python "$PROJECT_ROOT/scripts/09_littoral_constrained_rotation_quadrupole_inverse.py" >> "$LOG_FILE" 2>&1
echo "---------------------------------------------------" >> "$LOG_FILE"
python "$PROJECT_ROOT/scripts/10_littoral_mach_path_consistency.py" >> "$LOG_FILE" 2>&1
echo "---------------------------------------------------" >> "$LOG_FILE"
python "$PROJECT_ROOT/scripts/11_littoral_conditioned_mach_discrimination.py" >> "$LOG_FILE" 2>&1
echo "---------------------------------------------------" >> "$LOG_FILE"
date >> "$LOG_FILE"
echo "---------------------------------------------------" >> "$LOG_FILE"
