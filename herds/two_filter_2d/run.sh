python3 $1.py; air-opt --aten-to-xten $1.mlir > $1.air.mlir; air-opt --air-name-layers $1.air.mlir > $1.air_named.mlir; aten-opt --air-expand-graph $1.air_named.mlir