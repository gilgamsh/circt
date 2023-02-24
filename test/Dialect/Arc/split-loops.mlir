// RUN: circt-opt %s --arc-split-loops | FileCheck %s


// CHECK-LABEL: arc.define @Arc_split_0(%arg0: i4, %arg1: i4)
// CHECK-NEXT:    %0 = comb.add %arg0, %arg1
// CHECK-NEXT:    arc.output %0
// CHECK-NEXT:  }

// CHECK-LABEL: arc.define @Arc_split_1(%arg0: i4)
// CHECK-NEXT:    %0 = comb.mul %arg0, %arg0
// CHECK-NEXT:    arc.output %0
// CHECK-NEXT:  }

// CHECK-NOT: arc.define @Arc(
arc.define @Arc(%arg0: i4, %arg1: i4, %arg2: i4) -> (i4, i4) {
  %0 = comb.add %arg0, %arg1 : i4
  %1 = comb.mul %arg2, %arg2 : i4
  arc.output %0, %1 : i4, i4
}

// CHECK-LABEL: hw.module @Foo(
hw.module @Foo(%i0: i4, %i1: i4, %i2: i4) -> (z: i4, a: i4) {
  // CHECK-NEXT: %0 = arc.state @Arc_split_0(%i0, %i1)
  // CHECK-NEXT: %1 = arc.state @Arc_split_1(%i2)
  // CHECK-NEXT: hw.output %0, %1
  %0, %1 = arc.state @Arc(%i0, %i1, %i2) lat 0 : (i4, i4, i4) -> (i4, i4)
  hw.output %0, %1 : i4, i4
}
// CHECK-NEXT: }
