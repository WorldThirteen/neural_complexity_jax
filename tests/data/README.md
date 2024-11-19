# How data was obtained

To test for parity of the neural complexity calculation, with the original implementation in C++, the following steps were taken:

1. Data was obtained from the Open Source version of PolyWorld, which is available at https://github.com/polyworld/polyworld.
2. The hello world example was run to generate the data. The config was modified with `RecordNeuralComplexityFiles 1` to enable the recording of neural complexity.
3. 2 random brain activity files were picked and analyzed by the compiled `CalcComplexity` util in polyworld.
4. Complexity of all neurons was calculated for complexity, so target complexity was obtained using commands `./bin/CalcComplexity brainfunction <brainfunctionfile> A1`, `./bin/CalcComplexity brainfunction <brainfunctionfile> A0`
5. The same brain activity files were analyzed using the Python implementation of the neural complexity algorithm, they were processed using `../utils/convert_polyworld_activity.py` script.

