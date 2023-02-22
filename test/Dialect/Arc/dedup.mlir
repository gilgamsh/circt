// RUN: circt-opt %s --arc-dedup | FileCheck %s

// CHECK-LABEL: arc.define @SimpleA
arc.define @SimpleA(%arg0: i4, %arg1: i4) -> i4 {
  %0 = comb.and %arg0, %arg1 {Simple} : i4
  arc.output %0 : i4
}

// CHECK-NOT: arc.define @SimpleB
arc.define @SimpleB(%arg0: i4, %arg1: i4) -> i4 {
  %0 = comb.and %arg0, %arg1 {Simple} : i4
  arc.output %0 : i4
}

// CHECK-LABEL: hw.module @Simple
hw.module @Simple(%clock: i1, %x: i4, %y: i4) {
  // CHECK-NEXT: arc.state @SimpleA(%x, %y)
  // CHECK-NEXT: arc.state @SimpleA(%y, %x)
  %0 = arc.state @SimpleA(%x, %y) lat 0 : (i4, i4) -> i4
  %1 = arc.state @SimpleB(%y, %x) lat 0 : (i4, i4) -> i4
  // CHECK-NEXT: hw.output
}
// CHECK-NEXT: }


// CHECK-LABEL: arc.define @MismatchA
arc.define @MismatchA(%arg0: i4, %arg1: i4) -> i4 {
  %0 = comb.or %arg0, %arg1 {Mismatch} : i4
  arc.output %0 : i4
}

// CHECK-LABEL: arc.define @MismatchB
arc.define @MismatchB(%arg0: i4, %arg1: i4) -> i4 {
  %0 = comb.xor %arg0, %arg1 {Mismatch} : i4
  arc.output %0 : i4
}

// CHECK-LABEL: hw.module @Mismatch
hw.module @Mismatch(%clock: i1, %x: i4, %y: i4) {
  // CHECK-NEXT: arc.state @MismatchA(%x, %y)
  // CHECK-NEXT: arc.state @MismatchB(%y, %x)
  %0 = arc.state @MismatchA(%x, %y) lat 0 : (i4, i4) -> i4
  %1 = arc.state @MismatchB(%y, %x) lat 0 : (i4, i4) -> i4
  // CHECK-NEXT: hw.output
}
// CHECK-NEXT: }


// CHECK-LABEL: arc.define @OutlineConstA
// CHECK-SAME:    %arg0: i4
// CHECK-SAME:    %arg1: i4
arc.define @OutlineConstA(%arg0: i4) -> i4 {
  // CHECK-NEXT: comb.and %arg0, %arg1
  // CHECK-NEXT: arc.output
  %c3_i4 = hw.constant 3 : i4
  %0 = comb.and %arg0, %c3_i4 {OutlineConst} : i4
  arc.output %0 : i4
}
// CHECK-NEXT: }

// CHECK-NOT: arc.define @OutlineConstB
arc.define @OutlineConstB(%arg0: i4) -> i4 {
  %c7_i4 = hw.constant 7 : i4
  %0 = comb.and %arg0, %c7_i4 {OutlineConst} : i4
  arc.output %0 : i4
}

// CHECK-LABEL: hw.module @OutlineConst
hw.module @OutlineConst(%clock: i1, %x: i4, %y: i4) {
  // CHECK-DAG: %c3_i4 = hw.constant 3 : i4
  // CHECK-DAG: %c7_i4 = hw.constant 7 : i4
  // CHECK-DAG: arc.state @OutlineConstA(%x, %c3_i4)
  // CHECK-DAG: arc.state @OutlineConstA(%y, %c7_i4)
  %0 = arc.state @OutlineConstA(%x) lat 0 : (i4) -> i4
  %1 = arc.state @OutlineConstB(%y) lat 0 : (i4) -> i4
  // CHECK-NEXT: hw.output
}
// CHECK-NEXT: }


// CHECK-LABEL: arc.define @OutlineNonUniformConstsA
// CHECK-SAME:    %arg0: i4
// CHECK-SAME:    %arg1: i4
arc.define @OutlineNonUniformConstsA(%arg0: i4) -> i4 {
  // CHECK-NEXT: comb.mul %arg0, %arg1
  // CHECK-NEXT: arc.output
  %c3_i4 = hw.constant 3 : i4
  %0 = comb.mul %arg0, %c3_i4 {OutlineNonUniformConsts} : i4
  arc.output %0 : i4
}
// CHECK-NEXT: }

// CHECK-NOT: arc.define @OutlineNonUniformConstsB
arc.define @OutlineNonUniformConstsB(%arg0: i4) -> i4 {
  %c7_i4 = hw.constant 7 : i4
  %0 = comb.mul %c7_i4, %arg0 {OutlineNonUniformConsts} : i4
  arc.output %0 : i4
}

// CHECK-NOT: arc.define @OutlineNonUniformConstsC
arc.define @OutlineNonUniformConstsC() -> i4 {
  %c5_i4 = hw.constant 5 : i4
  %c4_i4 = hw.constant 4 : i4
  %0 = comb.mul %c5_i4, %c4_i4 {OutlineNonUniformConsts} : i4
  arc.output %0 : i4
}

// CHECK-NOT: arc.define @OutlineNonUniformConstsD
arc.define @OutlineNonUniformConstsD(%arg0: i4, %arg1: i4) -> i4 {
  %0 = comb.mul %arg0, %arg1 {OutlineNonUniformConsts} : i4
  arc.output %0 : i4
}

// CHECK-LABEL: hw.module @OutlineNonUniformConsts
hw.module @OutlineNonUniformConsts(%clock: i1, %x: i4) {
  // CHECK-DAG: %c3_i4 = hw.constant 3 : i4
  // CHECK-DAG: %c7_i4 = hw.constant 7 : i4
  // CHECK-DAG: %c5_i4 = hw.constant 5 : i4
  // CHECK-DAG: %c4_i4 = hw.constant 4 : i4
  // CHECK-DAG: arc.state @OutlineNonUniformConstsA(%x, %c3_i4)
  // CHECK-DAG: arc.state @OutlineNonUniformConstsA(%c7_i4, %x)
  // CHECK-DAG: arc.state @OutlineNonUniformConstsA(%c5_i4, %c4_i4)
  // CHECK-DAG: arc.state @OutlineNonUniformConstsA(%x, %x)
  %0 = arc.state @OutlineNonUniformConstsA(%x) lat 0 : (i4) -> i4
  %1 = arc.state @OutlineNonUniformConstsB(%x) lat 0 : (i4) -> i4
  %2 = arc.state @OutlineNonUniformConstsC() lat 0 : () -> i4
  %3 = arc.state @OutlineNonUniformConstsD(%x, %x) lat 0 : (i4, i4) -> i4
  // CHECK-NEXT: hw.output
}
// CHECK-NEXT: }


// CHECK-LABEL: arc.define @SplitArgumentsA
// CHECK-SAME:    %arg0: i4
// CHECK-SAME:    %arg1: i4
arc.define @SplitArgumentsA(%arg0: i4) -> i4 {
  // CHECK-NEXT: comb.and %arg0, %arg1
  // CHECK-NEXT: arc.output
  %0 = comb.and %arg0, %arg0 {SplitArguments} : i4
  arc.output %0 : i4
}
// CHECK-NEXT: }

// CHECK-NOT: arc.define @SplitArgumentsB
arc.define @SplitArgumentsB(%arg0: i4, %arg1: i4) -> i4 {
  %0 = comb.and %arg0, %arg1 {SplitArguments} : i4
  arc.output %0 : i4
}

// CHECK-NOT: arc.define @SplitArgumentsC
arc.define @SplitArgumentsC(%arg0: i4, %arg1: i4) -> i4 {
  %0 = comb.and %arg1, %arg0 {SplitArguments} : i4
  arc.output %0 : i4
}

// CHECK-LABEL: hw.module @SplitArguments
hw.module @SplitArguments(%clock: i1, %x: i4, %y: i4) {
  // CHECK-DAG: arc.state @SplitArgumentsA(%x, %x)
  // CHECK-DAG: arc.state @SplitArgumentsA(%x, %y)
  // CHECK-DAG: arc.state @SplitArgumentsA(%y, %x)
  %0 = arc.state @SplitArgumentsA(%x) lat 0 : (i4) -> i4
  %1 = arc.state @SplitArgumentsB(%x, %y) lat 0 : (i4, i4) -> i4
  %2 = arc.state @SplitArgumentsC(%x, %y) lat 0 : (i4, i4) -> i4
  // CHECK-NEXT: hw.output
}
// CHECK-NEXT: }



// CHECK-LABEL: arc.define @WeirdSplitArgumentsA
// CHECK-SAME:    %arg0: i4
// CHECK-SAME:    %arg1: i4
// CHECK-SAME:    %arg2: i4
// CHECK-SAME:    %arg3: i4
arc.define @WeirdSplitArgumentsA(%arg0: i4, %arg1: i4) -> i4 {
  // CHECK-NEXT: comb.and %arg0, %arg0, %arg1, %arg2, %arg3, %arg3
  // CHECK-NEXT: arc.output
  %0 = comb.and %arg0, %arg0, %arg0, %arg0, %arg1, %arg1 {WeirdSplitArguments} : i4
  arc.output %0 : i4
}
// CHECK-NEXT: }

// CHECK-NOT: arc.define @WeirdSplitArgumentsB
arc.define @WeirdSplitArgumentsB(%arg0: i4, %arg1: i4) -> i4 {
  %0 = comb.and %arg0, %arg0, %arg0, %arg1, %arg1, %arg1 {WeirdSplitArguments} : i4
  arc.output %0 : i4
}

// CHECK-NOT: arc.define @WeirdSplitArgumentsC
arc.define @WeirdSplitArgumentsC(%arg0: i4, %arg1: i4) -> i4 {
  %0 = comb.and %arg0, %arg0, %arg1, %arg0, %arg1, %arg1 {WeirdSplitArguments} : i4
  arc.output %0 : i4
}

// CHECK-LABEL: hw.module @WeirdSplitArguments
hw.module @WeirdSplitArguments(%clock: i1, %x: i4, %y: i4) {
  // CHECK-DAG: arc.state @WeirdSplitArgumentsA(%x, %x, %x, %y)
  // CHECK-DAG: arc.state @WeirdSplitArgumentsA(%x, %x, %y, %y)
  // CHECK-DAG: arc.state @WeirdSplitArgumentsA(%x, %y, %x, %y)
  %0 = arc.state @WeirdSplitArgumentsA(%x, %y) lat 0 : (i4, i4) -> i4
  %1 = arc.state @WeirdSplitArgumentsB(%x, %y) lat 0 : (i4, i4) -> i4
  %2 = arc.state @WeirdSplitArgumentsC(%x, %y) lat 0 : (i4, i4) -> i4
  // CHECK-NEXT: hw.output
}
// CHECK-NEXT: }


// CHECK-LABEL: arc.define @VariadicDiffsDontDedupA
arc.define @VariadicDiffsDontDedupA(%arg0: i4, %arg1: i4, %arg2: i4) -> i4 {
  %0 = comb.and %arg0, %arg1, %arg2 {VariadicDiffsDontDedup} : i4
  arc.output %0 : i4
}

// CHECK-LABEL: arc.define @VariadicDiffsDontDedupB
arc.define @VariadicDiffsDontDedupB(%arg0: i4, %arg1: i4) -> i4 {
  %0 = comb.and %arg0, %arg1 {VariadicDiffsDontDedup} : i4
  arc.output %0 : i4
}

// CHECK-LABEL: hw.module @VariadicDiffsDontDedup
hw.module @VariadicDiffsDontDedup(%clock: i1, %x: i4, %y: i4, %z: i4) {
  // CHECK-DAG: arc.state @VariadicDiffsDontDedupA(%x, %y, %z)
  // CHECK-DAG: arc.state @VariadicDiffsDontDedupB(%x, %y)
  %0 = arc.state @VariadicDiffsDontDedupA(%x, %y, %z) lat 0 : (i4, i4, i4) -> i4
  %1 = arc.state @VariadicDiffsDontDedupB(%x, %y) lat 0 : (i4, i4) -> i4
  // CHECK-NEXT: hw.output
}
// CHECK-NEXT: }
