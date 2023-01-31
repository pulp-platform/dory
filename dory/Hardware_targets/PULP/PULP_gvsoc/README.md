# NEW HARDWARE TARGET NAMES
Names of supported ISA extensions targeted by DORY changed: now you have to specify the name of the ISA extension in the --optional argument of the network_generator.

## Available targets
- "8bit" - former "8bit", RISCY + PULP-NN kernels
- "xpulpv2" - RISCY + PULP-NN-MIXED kernels (also SW mixed-precision)
- "xpulpnn" - XpulpNN (Mac&Load) + PULP-NN-MIXED kenels (SW mixed-precision Mac&Load)
- "xpulpnn-mixed" - XpulpNN-mixed (status-based mixed-precision Mac&Load) + PULP-NN-MIXED kernels (HW mixed-precision Mac&Load)

## WARNING!!
### For simulations on GVSOC of XpulpNN-mixed based networks a specific [version](https://github.com/alenad95/pulp-sdk/tree/ri5cy_nn_mix_integration) of the PULP-SDK is required since it's the only that supports these ISA extensions.