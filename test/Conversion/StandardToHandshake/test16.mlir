// NOTE: Assertions have been autogenerated by utils/update_mlir_test_checks.py
// RUN: circt-opt -lower-std-to-handshake %s | FileCheck %s
func @affine_apply_ceildiv(%arg0: index) -> index {
// CHECK:       module {

// CHECK-LABEL:   handshake.func @affine_apply_ceildiv(
// CHECK-SAME:                                         %[[VAL_0:.*]]: index, %[[VAL_1:.*]]: none, ...) -> (index, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "outCtrl"]} {
// CHECK:           %[[VAL_2:.*]] = "handshake.merge"(%[[VAL_0]]) : (index) -> index
// CHECK:           %[[VAL_3:.*]]:3 = "handshake.fork"(%[[VAL_2]]) {control = false} : (index) -> (index, index, index)
// CHECK:           %[[VAL_4:.*]]:4 = "handshake.fork"(%[[VAL_1]]) {control = true} : (none) -> (none, none, none, none)
// CHECK:           %[[VAL_5:.*]] = "handshake.constant"(%[[VAL_4]]#2) {value = 42 : index} : (none) -> index
// CHECK:           %[[VAL_6:.*]] = "handshake.constant"(%[[VAL_4]]#1) {value = 0 : index} : (none) -> index
// CHECK:           %[[VAL_7:.*]]:3 = "handshake.fork"(%[[VAL_6]]) {control = false} : (index) -> (index, index, index)
// CHECK:           %[[VAL_8:.*]] = "handshake.constant"(%[[VAL_4]]#0) {value = 1 : index} : (none) -> index
// CHECK:           %[[VAL_9:.*]]:2 = "handshake.fork"(%[[VAL_8]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_10:.*]] = arith.cmpi sle, %[[VAL_3]]#2, %[[VAL_7]]#0 : index
// CHECK:           %[[VAL_11:.*]]:2 = "handshake.fork"(%[[VAL_10]]) {control = false} : (i1) -> (i1, i1)
// CHECK:           %[[VAL_12:.*]] = arith.subi %[[VAL_7]]#1, %[[VAL_3]]#1 : index
// CHECK:           %[[VAL_13:.*]] = arith.subi %[[VAL_3]]#0, %[[VAL_9]]#0 : index
// CHECK:           %[[VAL_14:.*]] = select %[[VAL_11]]#1, %[[VAL_12]], %[[VAL_13]] : index
// CHECK:           %[[VAL_15:.*]] = arith.divsi %[[VAL_14]], %[[VAL_5]] : index
// CHECK:           %[[VAL_16:.*]]:2 = "handshake.fork"(%[[VAL_15]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_17:.*]] = arith.subi %[[VAL_7]]#2, %[[VAL_16]]#1 : index
// CHECK:           %[[VAL_18:.*]] = arith.addi %[[VAL_16]]#0, %[[VAL_9]]#1 : index
// CHECK:           %[[VAL_19:.*]] = select %[[VAL_11]]#0, %[[VAL_17]], %[[VAL_18]] : index
// CHECK:           handshake.return %[[VAL_19]], %[[VAL_4]]#3 : index, none
// CHECK:         }
// CHECK:       }

    %c42 = arith.constant 42 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = arith.cmpi sle, %arg0, %c0 : index
    %1 = arith.subi %c0, %arg0 : index
    %2 = arith.subi %arg0, %c1 : index
    %3 = select %0, %1, %2 : index
    %4 = arith.divsi %3, %c42 : index
    %5 = arith.subi %c0, %4 : index
    %6 = arith.addi %4, %c1 : index
    %7 = select %0, %5, %6 : index
    return %7 : index
  }
