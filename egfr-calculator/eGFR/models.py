"""
eGFR/models.py — Mechanistic eGFR Equations

Implements the three primary eGFR estimation equations:
  - CKD-EPI 2021 (race-free creatinine-based)
  - MDRD (4-variable, IDMS-traceable)
  - Cockcroft-Gault (creatinine clearance for drug dosing)

Each function accepts standard clinical inputs and returns estimated GFR
(or CrCl for Cockcroft-Gault) in mL/min/1.73 m² (or mL/min).
"""
