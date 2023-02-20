// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: arc.define @Foo
arc.define @Foo(%arg0: i42, %arg1: i9) -> (i42, i9) {
  // CHECK: arc.state @Bar(%arg0) lat 0 : (i42) -> i42
  %0 = arc.state @Bar(%arg0) lat 0 : (i42) -> i42

  // CHECK: arc.output %arg0, %arg1 : i42, i9
  arc.output %arg0, %arg1 : i42, i9
}

arc.define @Bar(%arg0: i42) -> i42 {
  arc.output %arg0 : i42
}

// CHECK-LABEL: hw.module @Module
hw.module @Module(%clock: i1, %enable: i1, %a: i42, %b: i9) {
  // CHECK: arc.state @Foo(%a, %b) clock %clock lat 1 : (i42, i9) -> (i42, i9)
  arc.state @Foo(%a, %b) clock %clock lat 1 : (i42, i9) -> (i42, i9)

  // CHECK: arc.state @Foo(%a, %b) clock %clock enable %enable lat 1 : (i42, i9) -> (i42, i9)
  arc.state @Foo(%a, %b) clock %clock enable %enable lat 1 : (i42, i9) -> (i42, i9)
}

// CHECK-LABEL: @LookupTable(%arg0: i32, %arg1: i8)
arc.define @LookupTable(%arg0: i32, %arg1: i8) -> () {
  // CHECK-NEXT: %{{.+}} = arc.lut() : () -> i32 {
  // CHECK-NEXT:   %c0_i32 = hw.constant 0 : i32
  // CHECK-NEXT:   arc.output %c0_i32 : i32
  // CHECK-NEXT: }
  %0 = arc.lut () : () -> i32 {
    ^bb0():
      %0 = hw.constant 0 : i32
      arc.output %0 : i32
  }
  // CHECK-NEXT: %{{.+}} = arc.lut(%arg1, %arg0) : (i8, i32) -> i32 {
  // CHECK-NEXT: ^bb0(%arg2: i8, %arg3: i32):
  // CHECK-NEXT:   arc.output %arg3 : i32
  // CHECK-NEXT: }
  %1 = arc.lut (%arg1, %arg0) : (i8, i32) -> i32 {
    ^bb0(%arg2: i8, %arg3: i32):
      arc.output %arg3 : i32
  }
  arc.output
}
