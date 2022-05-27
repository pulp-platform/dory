#include <element_wise_sum.h>

// Warning: we already assume here that A and B have the same shape!
int32_t element_wise_sum(const void *inputA, const void *inputB, void *output,
                         SomaShape shape,
                         uint32_t precision) {
    debug_print(1,"%s\n","Begin SOMA Wrapped EWS");
    // First Step: Enabling the SOMA accelerator.
    soma_enable();
    // TODO manually unpacking here now
    uint32_t w = shape.shape[0];
    uint32_t h = shape.shape[1];
    uint32_t c = shape.shape[2];
    // Second step: Preparing the computation.
    //
    // Here a GPU program would build the kernels for the CPU, instead we create
    // a SomaOperation with the desired instruction. We are still providing the
    // shape of the tensors and the precision to the function. For
    // "shift_fixed_point" I used 0 because I don't know which value is
    // expected.
    SomaOperation ews_operation = soma_element_wise_sum(w, h, c, 8, 0);

    // Fifth step: Preparing the arguments.
    //
    // We tell SOMA which tensors it should use as arguments. There could be
    // SomaTensors currently on the L1 but not involved in the computation,
    // maybe because they will be reused for a residual layer.
    //
    // Setting the first input, so kind=0 which means first operand.
    soma_set_argument(&ews_operation, &tensorA, 0);
    // Setting the second input, so kind=1.
    soma_set_argument(&ews_operation, &tensorB, 1);
    // Setting the output, so kind=3 which means that this is the output and
    // that it should be copied back to L1 and L2. kind=2 would means that the
    // output only goes back to L1.
    soma_set_argument(&ews_operation, &tensor_output, 3);
    // Sixth step: Starting the computation.
    //
    // The SomaOperation has been prepared before, now we only need to start it.
    soma_enqueue(&ews_operation);

    // Last step: Disabling the SOMA accelerator.
    soma_disable();
    debug_print(1,"%s\n","Successfully performed SOMA Wrapped EWS");
    // Execution was a success.
    return 0;
}