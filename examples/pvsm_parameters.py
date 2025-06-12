from FEelMRI.Parameters import PVSMParser

# Given a .pvsm file, extract the lengths of the box source and the transform parameters. The results should be in the units specified by the user and match those observed in the ParaView GUI when loading the .pvsm file
parser = PVSMParser("planning/phase_contrast.pvsm", 
                    box_name="Box1", 
                    transform_name="Transform1",
                    length_units="cm")

print("FOV Dimensions:", parser.FOV)
print("Transform Position:", parser.LOC)
print("Transform Rotation:", parser.Rotation)