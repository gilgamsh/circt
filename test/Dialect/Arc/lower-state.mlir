// RUN: circt-opt %s --arc-lower-state | FileCheck %s

// CHECK-LABEL: arc.model "Empty" {
// CHECK-NEXT:  ^bb0(%arg0: !arc.storage):
// CHECK-NEXT:  }
hw.module @Empty() {
}

// CHECK-LABEL: arc.model "InputsAndOutputs" {
hw.module @InputsAndOutputs(%a: i42, %b: i17) -> (c: i42, d: i17) {
  %0 = comb.add %a, %a : i42
  %1 = comb.add %b, %b : i17
  hw.output %0, %1 : i42, i17
  // CHECK-NEXT: ([[PTR:%.+]]: !arc.storage):
  // CHECK-NEXT: [[INA:%.+]] = arc.root_input "a", [[PTR]] : (!arc.storage) -> !arc.state<i42>
  // CHECK-NEXT: [[INB:%.+]] = arc.root_input "b", [[PTR]] : (!arc.storage) -> !arc.state<i17>
  // CHECK-NEXT: arc.passthrough {
  // CHECK-NEXT:   [[A:%.+]] = arc.state_read [[INA]] : <i42>
  // CHECK-NEXT:   [[TMP:%.+]] = comb.add [[A]], [[A]] : i42
  // CHECK-NEXT:   arc.state_write [[OUTA:%.+]] = [[TMP]] : <i42>
  // CHECK-NEXT:   [[B:%.+]] = arc.state_read [[INB]] : <i17>
  // CHECK-NEXT:   [[TMP:%.+]] = comb.add [[B]], [[B]] : i17
  // CHECK-NEXT:   arc.state_write [[OUTB:%.+]] = [[TMP]] : <i17>
  // CHECK-NEXT: }
  // CHECK-NEXT: [[OUTA]] = arc.root_output "c", [[PTR]] : (!arc.storage) -> !arc.state<i42>
  // CHECK-NEXT: [[OUTB]] = arc.root_output "d", [[PTR]] : (!arc.storage) -> !arc.state<i17>
}

// CHECK-LABEL: arc.model "State" {
hw.module @State(%clk: i1, %en: i1) {
  %gclk = arc.clock_gate %clk, %en
  %3 = arc.state @DummyArc(%6) clock %clk lat 1 : (i42) -> i42
  %4 = arc.state @DummyArc(%5) clock %gclk lat 1 : (i42) -> i42
  %5 = comb.add %3, %3 : i42
  %6 = comb.add %4, %4 : i42
  // CHECK-NEXT: ([[PTR:%.+]]: !arc.storage):
  // CHECK-NEXT: [[INCLK:%.+]] = arc.root_input "clk", [[PTR]] : (!arc.storage) -> !arc.state<i1>
  // CHECK-NEXT: [[CLK:%.+]] = arc.state_read [[INCLK]] : <i1>
  // CHECK-NEXT: [[INEN:%.+]] = arc.root_input "en", [[PTR]] : (!arc.storage) -> !arc.state<i1>
  // CHECK-NEXT: arc.clock_tree [[CLK]] {
  // CHECK-NEXT:   [[TMP0:%.+]] = arc.state_read [[S1:%.+]] : <i42>
  // CHECK-NEXT:   [[TMP1:%.+]] = comb.add [[TMP0]], [[TMP0]]
  // CHECK-NEXT:   [[TMP2:%.+]] = arc.state @DummyArc([[TMP1]]) lat 0 : (i42) -> i42
  // CHECK-NEXT:   arc.state_write [[S0:%.+]] = [[TMP2]] : <i42>
  // CHECK-NEXT:   [[TMP0:%.+]] = arc.state_read [[S0]] : <i42>
  // CHECK-NEXT:   [[TMP1:%.+]] = comb.add [[TMP0]], [[TMP0]]
  // CHECK-NEXT:   [[TMP2:%.+]] = arc.state @DummyArc([[TMP1]]) lat 0 : (i42) -> i42
  // CHECK-NEXT:   [[EN:%.+]] = arc.state_read [[INEN]] : <i1>
  // CHECK-NEXT:   arc.state_write [[S1]] = [[TMP2]] if [[EN]] : <i42>
  // CHECK-NEXT: }
  // CHECK-NEXT: [[S0]] = arc.alloc_state [[PTR]] : (!arc.storage) -> !arc.state<i42>
  // CHECK-NEXT: [[S1]] = arc.alloc_state [[PTR]] : (!arc.storage) -> !arc.state<i42>
}

// CHECK-LABEL: arc.model "State2" {
hw.module @State2(%clk: i1) {
  %3 = arc.state @DummyArc(%3) clock %clk lat 1 : (i42) -> i42
  %4 = arc.state @DummyArc(%4) clock %clk lat 1 : (i42) -> i42
  // CHECK-NEXT: ^bb
  // CHECK-NEXT: %in_clk = arc.root_input "clk", %arg0 : (!arc.storage) -> !arc.state<i1>
  // CHECK-NEXT: [[CLK:%.+]] = arc.state_read %in_clk : <i1>
  // CHECK-NEXT: arc.clock_tree [[CLK]] {
  // CHECK-NEXT:   [[TMP0:%.+]] = arc.state_read [[S0:%.+]] : <i42>
  // CHECK-NEXT:   [[TMP1:%.+]] = arc.state @DummyArc([[TMP0]]) lat 0 : (i42) -> i42
  // CHECK-NEXT:   arc.state_write [[S0]] = [[TMP1]] : <i42>
  // CHECK-NEXT:   [[TMP2:%.+]] = arc.state_read [[S1:%.+]] : <i42>
  // CHECK-NEXT:   [[TMP3:%.+]] = arc.state @DummyArc([[TMP2]]) lat 0 : (i42) -> i42
  // CHECK-NEXT:   arc.state_write [[S1]] = [[TMP3]] : <i42>
  // CHECK-NEXT: }
  // CHECK-NEXT: [[S0]] = arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i42>
  // CHECK-NEXT: [[S1]] = arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i42>
}

arc.define @DummyArc(%arg0: i42) -> i42 {
  arc.output %arg0 : i42
}

// CHECK-LABEL: arc.model "MemoryWriteDependencyUpdates"
hw.module @MemoryWriteDependencyUpdates(%clk0: i1, %clk1: i1) {
  %true = hw.constant true
  %c0_i2 = hw.constant 0 : i2
  %c1_i2 = hw.constant 1 : i2
  %c9001_i42 = hw.constant 9001 : i42
  %mem = arc.memory <4 x i42>
  %read0 = arc.memory_read %mem[%c0_i2], %clk0, %true : <4 x i42>, i2
  %read1 = arc.memory_read %mem[%c1_i2], %clk1, %true : <4 x i42>, i2
  arc.memory_write %mem[%c0_i2], %clk0, %true, %c9001_i42 (reads %read0, %read1 : i42, i42) : <4 x i42>, i2
  arc.memory_write %mem[%c1_i2], %clk1, %true, %c9001_i42 (reads %read0, %read1 : i42, i42) : <4 x i42>, i2
  // CHECK-NEXT: ([[PTR:%.+]]: !arc.storage):
  // CHECK-NEXT: [[INCLK0:%.+]] = arc.root_input "clk0", [[PTR]] : (!arc.storage) -> !arc.state<i1>
  // CHECK-NEXT: [[INCLK1:%.+]] = arc.root_input "clk1", [[PTR]] : (!arc.storage) -> !arc.state<i1>
  // CHECK-NEXT: [[MEM:%.+]] = arc.alloc_memory [[PTR]] : (!arc.storage) -> !arc.memory<4 x i42>
  // CHECK-NEXT: [[CLK0:%.+]] = arc.state_read [[INCLK0]] : <i1>
  // CHECK-NEXT: arc.clock_tree [[CLK0]] {
  // CHECK:        [[CLK:%.+]] = arc.state_read [[INCLK0]]
  // CHECK-NEXT:   [[TMP:%.+]] = arc.memory_read [[MEM]][%c0_i2], [[CLK]], %true : <4 x i42>, i2
  //               Only one read should remain.
  // CHECK-NEXT:   arc.memory_write [[MEM]][%c0_i2], [[CLK]], %true, %c9001_i42 (reads [[TMP]] : i42) : <4 x i42>, i2
  // CHECK-NEXT: }
  // CHECK-NEXT: [[CLK1:%.+]] = arc.state_read [[INCLK1]] : <i1>
  // CHECK-NEXT: arc.clock_tree [[CLK1]] {
  // CHECK:        [[CLK:%.+]] = arc.state_read [[INCLK1]]
  // CHECK-NEXT:   [[TMP:%.+]] = arc.memory_read [[MEM]][%c1_i2], [[CLK]], %true : <4 x i42>, i2
  //               Only one read should remain.
  // CHECK-NEXT:   arc.memory_write [[MEM]][%c1_i2], [[CLK]], %true, %c9001_i42 (reads [[TMP]] : i42) : <4 x i42>, i2
  // CHECK-NEXT: }
}

// CHECK-LABEL:  arc.model "maskedMemoryWrite"
hw.module @maskedMemoryWrite(%clk: i1) {
  %true = hw.constant true
  %c0_i2 = hw.constant 0 : i2
  %c9001_i42 = hw.constant 9001 : i42
  %c1010_i42 = hw.constant 1010 : i42
  %mem = arc.memory <4 x i42>
  arc.memory_write %mem[%c0_i2], %clk, %true, %c9001_i42 mask (%c1010_i42 : i42) : <4 x i42>, i2
}
// CHECK:      %c9001_i42 = hw.constant 9001 : i42
// CHECK:      %c1010_i42 = hw.constant 1010 : i42
// CHECK:      [[RD:%.+]] = arc.memory_read [[MEM:%.+]][%c0_i2], [[CLK:%.+]], %true : <4 x i42>, i2
// CHECK:      %c-1_i42 = hw.constant -1 : i42
// CHECK:      [[NEG_MASK:%.+]] = comb.xor bin %c1010_i42, %c-1_i42 : i42
// CHECK:      [[OLD_MASKED:%.+]] = comb.and bin [[NEG_MASK]], [[RD]] : i42
// CHECK:      [[NEW_MASKED:%.+]] = comb.and bin %c1010_i42, %c9001_i42 : i42
// CHECK:      [[DATA:%.+]] = comb.or bin [[OLD_MASKED]], [[NEW_MASKED]] : i42
// CHECK:      arc.memory_write [[MEM]][%c0_i2], [[CLK]], %true, [[DATA]] : <4 x i42>, i2
