from pynestml.codegeneration.nest_code_generator_utils import NESTCodeGeneratorUtils
import pickle

module_name, neuron_model_name = NESTCodeGeneratorUtils.generate_code_for(str(snakemake.input))

with open(str(snakemake.output), 'wb') as file:
    pickle.dump((var1, var2), file)
