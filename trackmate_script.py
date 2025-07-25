#@ File (style='file') inFile
#@ File (style="save") xmlFile

import sys

from ij import IJ
from ij import WindowManager

from fiji.plugin.trackmate import Model
from fiji.plugin.trackmate import Settings
from fiji.plugin.trackmate import TrackMate
from fiji.plugin.trackmate import SelectionModel
from fiji.plugin.trackmate import Logger
from fiji.plugin.trackmate.detection import MaskDetectorFactory
from fiji.plugin.trackmate.tracking.jaqaman import SparseLAPTrackerFactory
import fiji.plugin.trackmate.features.FeatureFilter as FeatureFilter
from fiji.plugin.trackmate.io import TmXmlWriter


# We have to do the following to avoid errors with UTF8 chars generated in 
# TrackMate that will mess with our Fiji Jython.
reload(sys)
sys.setdefaultencoding('utf-8')

# Get currently selected image
# imp = WindowManager.getCurrentImage()
# imp = IJ.openImage('https://fiji.sc/samples/FakeTracks.tif')
imp = IJ.openImage(inFile.getAbsolutePath())


#----------------------------
# Create the model object now
#----------------------------

# Some of the parameters we configure below need to have
# a reference to the model at creation. So we create an
# empty model now.

model = Model()

# Send all messages to ImageJ log window.
model.setLogger(Logger.IJ_LOGGER)



#------------------------
# Prepare settings object
#------------------------

settings = Settings(imp)

# Configure detector - We use the Strings for the keys
settings.detectorFactory = MaskDetectorFactory()
settings.detectorSettings = {
    'TARGET_CHANNEL' : 1,
    'SIMPLIFY_CONTOURS' : False,
}  

# Configure tracker - We want to allow merges and fusions
settings.trackerFactory = SparseLAPTrackerFactory()
settings.trackerSettings = settings.trackerFactory.getDefaultSettings() # almost good enough
settings.trackerSettings['ALLOW_TRACK_SPLITTING'] = True
settings.trackerSettings['ALLOW_TRACK_MERGING'] = True

# Add ALL the feature analyzers known to TrackMate. They will 
# yield numerical features for the results, such as speed, mean intensity etc.
settings.addAllAnalyzers()


#-------------------
# Instantiate plugin
#-------------------

trackmate = TrackMate(model, settings)

#--------
# Process
#--------

ok = trackmate.checkInput()
if not ok:
    sys.exit(str(trackmate.getErrorMessage()))

ok = trackmate.process()
if not ok:
    sys.exit(str(trackmate.getErrorMessage()))


#-------------
# Write to XML
#-------------

writer = TmXmlWriter(xmlFile)
writer.appendModel(model)
writer.appendSettings(settings)
writer.writeToFile()