/*F***************************************************************************
 * This file is part of openSMILE.
 * 
 * Copyright (c) audEERING GmbH. All rights reserved.
 * See the file COPYING for details on license terms.
 ***************************************************************************E*/


/*  openSMILE component:


*/


#ifndef __CARFFSOURCE_HPP
#define __CARFFSOURCE_HPP

#include <core/smileCommon.hpp>
#include <core/dataSource.hpp>

#define COMPONENT_DESCRIPTION_CARFFSOURCE "This component reads WEKA ARFF files. The full ARFF format is not yet supported, but a simplified form, such as the files generated by the cArffSink component can be parsed and read. This component reads all (and only!!) 'numeric' or 'real' attributes from an ARFF file (WEKA file format) into the specified data memory level. Thereby each instance (i.e. one line in the arff file's data section) corresponds to one frame. The frame period is 0 by default (aperiodic level), use the 'period' option to change this and use a fixed period for each frame/instance. Automatic generation of frame timestamps from a 'timestamp' field in the Arff file is not yet supported."
#define COMPONENT_NAME_CARFFSOURCE "cArffSource"

class cArffSource : public cDataSource {
  private:
    int nAttr;
    int saveClassesAsMetadata;
    int lastNumeric;
    int useInstanceID;
    int strField;
    FILE *filehandle;
    const char *filename;
    int *field;
    int fieldNalloc;
    int nFields,nNumericFields;
    int eof;
    int skipClasses;
    long lineNr;
    size_t lineLen;
    char *origline;
    int skipFirst;
    bool readFrameLength_;
    bool readFrameTime_;
    int frameLengthNr_;
    int frameTimeNr_;

  protected:
    SMILECOMPONENT_STATIC_DECL_PR
    
    virtual void myFetchConfig() override;
    //virtual int myConfigureInstance() override;
    virtual int configureWriter(sDmLevelConfig &c) override;
    virtual int myFinaliseInstance() override;
    virtual eTickResult myTick(long long t) override;

    virtual int setupNewNames(long nEl=0) override;

  public:
    SMILECOMPONENT_STATIC_DECL
    
    cArffSource(const char *_name);

    virtual ~cArffSource();
};




#endif // __CARFFSOURCE_HPP
