// RUN: circt-opt %s --arc-lower-clocks-to-funcs --verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @Trivial_clock(%arg0: !arc.storage<42>) {
// CHECK-NEXT:    %c0_i9001 = hw.constant 0 : i9001
// CHECK-NEXT:    %true = hw.constant true
// CHECK-NEXT:    %0 = comb.mux %true, %c0_i9001, %c0_i9001 : i9001
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// CHECK-LABEL: func.func @Trivial_passthrough(%arg0: !arc.storage<42>) {
// CHECK-NEXT:    %c1_i9001 = hw.constant 1 : i9001
// CHECK-NEXT:    %true = hw.constant true
// CHECK-NEXT:    %0 = comb.mux %true, %c1_i9001, %c1_i9001 : i9001
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// CHECK-LABEL: arc.model "Trivial" {
// CHECK-NEXT:  ^bb0(%arg0: !arc.storage<42>):
// CHECK-NEXT:    %false = hw.constant false
// CHECK-NEXT:    func.call @Trivial_clock(%arg0) : (!arc.storage<42>) -> ()
// CHECK-NEXT:    func.call @Trivial_passthrough(%arg0) : (!arc.storage<42>) -> ()
// CHECK-NEXT:  }

arc.model "Trivial" {
^bb0(%arg0: !arc.storage<42>):
  %true = hw.constant true
  %false = hw.constant false
  arc.clock_tree %true {
    %c0_i9001 = hw.constant 0 : i9001
    %0 = comb.mux %true, %c0_i9001, %c0_i9001 : i9001
  }
  arc.passthrough {
    %c1_i9001 = hw.constant 1 : i9001
    %0 = comb.mux %true, %c1_i9001, %c1_i9001 : i9001
  }
}
