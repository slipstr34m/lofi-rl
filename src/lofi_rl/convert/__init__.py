"""SFZ to SF2 conversion for the shootout banks FluidSynth cannot load (docs/CONVERTER.md).

sfz: parser, ARIA evaluation at the controller defaults, region selection. mapping: regions to SF2 zones and the
fits. sf2: the byte-deterministic SoundFont 2.04 writer and lossless sample reading. banks: the six reduced bank
mappings. `python -m lofi_rl.convert` is the CLI.
"""
