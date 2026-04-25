#!/bin/sh
date > logs/geospatial.log
echo "---------------------------------------------------" >> logs/geospatial.log
python scripts/01_geospatial_gradient_scan.py >> logs/geospatial.log
echo "---------------------------------------------------" >> logs/geospatial.log
python scripts/02_geospatial_robust_validation.py >> logs/geospatial.log
echo "---------------------------------------------------" >> logs/geospatial.log
python scripts/03_littoral_balanced_geometry.py >> logs/geospatial.log
echo "---------------------------------------------------" >> logs/geospatial.log
python scripts/04_littoral_axis_hypothesis_tests.py >> logs/geospatial.log
echo "---------------------------------------------------" >> logs/geospatial.log
python scripts/05_littoral_spectral_decomposition.py >> logs/geospatial.log
echo "---------------------------------------------------" >> logs/geospatial.log
python scripts/06_littoral_quadrupole_validation.py >> logs/geospatial.log
echo "---------------------------------------------------" >> logs/geospatial.log
python scripts/07_littoral_rotation_inverse.py >> logs/geospatial.log
echo "---------------------------------------------------" >> logs/geospatial.log
python scripts/08_littoral_joint_rotation_quadrupole_inverse.py >> logs/geospatial.log
echo "---------------------------------------------------" >> logs/geospatial.log
python scripts/09_littoral_constrained_rotation_quadrupole_inverse.py >> logs/geospatial.log
echo "---------------------------------------------------" >> logs/geospatial.log
python scripts/10_littoral_mach_path_consistency.py >> logs/geospatial.log
echo "---------------------------------------------------" >> logs/geospatial.log
python scripts/11_littoral_conditioned_mach_discrimination.py >> logs/geospatial.log
echo "---------------------------------------------------" >> logs/geospatial.log
date > logs/geospatial.log
echo "---------------------------------------------------" >> logs/geospatial.log
