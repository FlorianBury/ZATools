####### Warning: put most recent tags first! ###### 
analysis_tags = [
        'v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a'  #JEC splitting
        #'v6.1.0+80X_ZAAnalysis_2017-12-12.v0-3-g6e23962' # unblind 1/10 of data (tag used for the 3rd version of the AN only for data)
        #'v6.1.0+80X_ZAAnalysis_2017-12-12.v0-2-gf03f531' # back to cut based id for electrons (tag used for the 3rd version of the AN)
        #'v6.1.0+80X_ZAAnalysis_2017-12-12.v0-1-g21432f5' # --> back to cut based id for electrons (wrong, iso cut still included)
        #'v6.1.0+80X_ZAAnalysis_2017-12-12.v0' # --> m_electron_mva_wp80_name bug fixed, JEC and JER as systematics, new trigger eff files for electrons from hww 
        #'v6.1.0+80X_ZAAnalysis_2017-11-10.v0' # --> m_electron_mva_wp80_name bug
        #'v6.0.0+80X_ZAAnalysis_2017-09-27.v1' --> cmva bug, no METsignificance
        #'v5.0.1+80X-7-g03c2b54_ZAAnalysis_Moriond2015-7-g08c899b' # --> electrons are good
        #'v5.0.1+80X-2-g909e9e2_ZAAnalysis_Moriond2015-5-g0d38378'
        #'v5.0.1+80X-2-g909e9e2_ZAAnalysis_Moriond2015-1-gd479ab9'
        ]

samples_dict = {}


# Data
samples_dict["Data"] = [
    'DoubleEG',
    'MuonEG',
    'DoubleMuon'
]

# DY NLO
samples_dict["DY_NLO"] = [
    'DYJetsToLL_M-10to50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8_extended_ext0_plus_ext1',
    'DYToLL_0J_13TeV-amcatnloFXFX-pythia8_extended_ext0_plus_ext1',
    'DYToLL_1J_13TeV-amcatnloFXFX-pythia8_extended_ext0_plus_ext1',
    'DYToLL_2J_13TeV-amcatnloFXFX-pythia8_extended_ext0_plus_ext1'
]

# TTBar
samples_dict["TTBar"] = [
    'TT_TuneCUETP8M2T4_13TeV-powheg-pythia8_extended_ext0_plus_ext1',
    'TTTo2L2Nu_13TeV-powheg_Summer16MiniAODv2'
]

# ZZ
samples_dict["ZZ"] = [
     'ZZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8',
     'ZZTo2L2Nu_13TeV_powheg_pythia8',
     'ZZTo4L_13TeV_powheg_pythia8'
]

# ZH
samples_dict["ZH"] = [
    'HZJ_HToWW_M125_13TeV_powheg_pythia8_Summer16MiniAODv2',
    'GluGluZH_HToWWTo2L2Nu_ZTo2L_M125_13TeV_powheg_pythia8_Summer16MiniAODv2',
    'ggZH_HToBB_ZToLL_M125_13TeV_powheg_pythia8_Summer16MiniAODv2',
    'ZH_HToBB_ZToLL_M125_13TeV_powheg_pythia8_Summer16MiniAODv2',
    'ggZH_HToBB_ZToNuNu_M125_13TeV_powheg_pythia8_Summer16MiniAODv2'
]

#WGamma
samples_dict["WGamma"] = [
    'WGToLNuG_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8_extended_ext0_plus_ext1_plus_ext2_plus_ext3'
]

# VV
samples_dict["VV"] = [
    # WW
    'WWToLNuQQ_13TeV-powheg_Summer16MiniAODv2',
    'WWTo2L2Nu_13TeV-powheg_Summer16MiniAODv2',
    'WWTo4Q_13TeV-powheg_Summer16MiniAODv2',
    # WZ
    'WZTo3LNu_TuneCUETP8M1_13TeV-powheg-pythia8_Summer16MiniAODv2',
    'WZTo1L3Nu_13TeV_amcatnloFXFX_madspin_pythia8_Summer16MiniAODv2',
    #'WZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8_Summer16MiniAODv2',
    'WZTo1L1Nu2Q_13TeV_amcatnloFXFX_madspin_pythia8_Summer16MiniAODv2',
    # WZZ
    'WZZ_TuneCUETP8M1_13TeV-amcatnlo-pythia8_Summer16MiniAODv2',
    # WWZ
    'WWZ_TuneCUETP8M1_13TeV-amcatnlo-pythia8_Summer16MiniAODv2',
    # WWW
    'WWW_TuneCUETP8M1_13TeV-amcatnlo-pythia8_Summer16MiniAODv2',
    # ZZZ
    'ZZZ_TuneCUETP8M1_13TeV-amcatnlo-pythia8_Summer16MiniAODv2'
]

# WJets 
samples_dict["WJets"] = [
    'WJetsToLNu_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8_Summer16MiniAODv2'
]

# TTV
samples_dict["TTV"] = [
    'TTZToQQ_TuneCUETP8M1_13TeV-amcatnlo-pythia8_Summer16MiniAODv2',
    'TTWJetsToQQ_TuneCUETP8M1_13TeV-amcatnloFXFX-madspin-pythia8_Summer16MiniAODv2',
    'TTWJetsToLNu_TuneCUETP8M1_13TeV-amcatnloFXFX-madspin-pythia8_Summer16MiniAODv2',
    'TTZToLLNuNu_M-10_TuneCUETP8M1_13TeV-amcatnlo-pythia8_Summer16MiniAODv2'
]

# TTH
samples_dict["TTH"] = [
    'ttHToNonbb_M125_TuneCUETP8M2_13TeV_powheg_pythia8_Summer16MiniAODv2',
    'ttHTobb_M125_TuneCUETP8M2_13TeV_powheg_pythia8_Summer16MiniAODv2'
]

# Single top
samples_dict["SingleTop"] = [
    'ST_tW_antitop_5f_noFullyHadronicDecays_13TeV-powheg_Summer16MiniAODv2',
    'ST_tW_top_5f_noFullyHadronicDecays_13TeV-powheg_Summer16MiniAODv2',
    'ST_s-channel_4f_leptonDecays_13TeV-amcatnlo_Summer16MiniAODv2',
    'ST_t-channel_top_4f_inclusiveDecays_13TeV-powheg-pythia8_Summer16MiniAODv2',
    'ST_t-channel_antitop_4f_inclusiveDecays_13TeV-powheg-pythia8_Summer16MiniAODv'
]

# QCD
samples_dict["QCD"] = [
    #'QCD_Pt-20to30_EMEnriched_TuneCUETP8M1_13TeV_pythia8_Summer16MiniAODv2',
    #'QCD_Pt-30to50_EMEnriched_TuneCUETP8M1_13TeV_pythia8_extended_ext0_plus_ext1',
    'QCD_Pt-50to80_EMEnriched_TuneCUETP8M1_13TeV_pythia8_extended_ext0_plus_ext1',
    'QCD_Pt-80to120_EMEnriched_TuneCUETP8M1_13TeV_pythia8_extended_ext0_plus_ext1',
    'QCD_Pt-120to170_EMEnriched_TuneCUETP8M1_13TeV_pythia8_extended_ext0_plus_ext1',
    'QCD_Pt-170to300_EMEnriched_TuneCUETP8M1_13TeV_pythia8_Summer16MiniAODv2',
    'QCD_Pt-300toInf_EMEnriched_TuneCUETP8M1_13TeV_pythia8_Faill15MiniAODv2',   #keep it?
    'QCD_Pt-20toInf_MuEnrichedPt15_TuneCUETP8M1_13TeV_pythia8_Summer16MiniAODv2'
]

## Signals
samples_dict["Signal"] = [
    'HToZATo2L2B_1000p00_200p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_1000p00_500p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_1000p00_50p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_132p00_30p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_132p00_37p34_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_143p44_30p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_143p44_37p34_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_143p44_46p48_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_157p77_30p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_157p77_37p34_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_157p77_46p48_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_157p77_57p85_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_173p52_30p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_173p52_37p34_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_173p52_46p48_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_173p52_57p85_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_173p52_72p01_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_190p85_30p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_190p85_37p34_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_190p85_46p48_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_190p85_57p85_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_190p85_71p28_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_190p85_86p78_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_200p00_100p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_200p00_50p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_209p90_104p53_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_209p90_30p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_209p90_37p34_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_209p90_46p48_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_209p90_57p71_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_209p90_71p15_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_209p90_86p79_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_230p77_102p72_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_230p77_123p89_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_230p77_30p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_230p77_37p10_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_230p77_45p88_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_230p77_56p73_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_230p77_69p78_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_230p77_85p09_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_250p00_100p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_250p00_50p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_261p40_102p99_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_261p40_124p53_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_261p40_150p50_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_261p40_37p10_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_261p40_45p88_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_261p40_56p73_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_261p40_69p66_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_261p40_85p10_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_296p10_120p82_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_296p10_145p93_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_296p10_176p02_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_296p10_30p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_296p10_36p79_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_296p10_45p12_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_296p10_55p33_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_296p10_67p65_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_296p10_82p40_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_296p10_99p90_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_300p00_100p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_300p00_200p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_300p00_50p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_335p40_120p39_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_335p40_145p06_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_335p40_174p55_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_335p40_209p73_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_335p40_30p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_335p40_36p79_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_335p40_45p12_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_335p40_55p33_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_335p40_67p54_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_335p40_82p14_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_335p40_99p61_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_379p00_118p81_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_379p00_143p08_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_379p00_171p71_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_379p00_205p76_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_379p00_246p30_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_379p00_30p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_379p00_36p63_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_379p00_44p72_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_379p00_54p59_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_379p00_66p57_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_379p00_80p99_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_379p00_98p26_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_442p63_113p53_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_442p63_135p44_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_442p63_161p81_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_442p63_193p26_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_442p63_230p49_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_442p63_274p57_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_442p63_30p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_442p63_327p94_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_442p63_36p64_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_442p63_44p76_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_442p63_54p67_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_442p63_66p49_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_442p63_80p03_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_442p63_95p27_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_500p00_100p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_500p00_200p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_500p00_300p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_500p00_400p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_500p00_50p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_516p94_109p30_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_516p94_128p58_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_516p94_151p69_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_516p94_179p35_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_516p94_212p14_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_516p94_250p63_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_516p94_296p65_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_516p94_30p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_516p94_352p61_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_516p94_36p47_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_516p94_423p96_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_516p94_44p34_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_516p94_53p90_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_516p94_65p52_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_516p94_78p52_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_516p94_93p12_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_609p21_116p29_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_609p21_135p66_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_609p21_158p41_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_609p21_185p18_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_609p21_216p52_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_609p21_253p68_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_609p21_298p01_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_609p21_30p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_609p21_34p86_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_609p21_351p22_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_609p21_40p51_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_609p21_417p76_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_609p21_47p08_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_609p21_505p93_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_609p21_54p71_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_609p21_63p58_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_609p21_85p86_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_609p21_99p78_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_650p00_50p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_717p96_116p19_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_717p96_157p56_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_717p96_183p48_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_717p96_213p73_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_717p96_249p34_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_717p96_291p34_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_717p96_30p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_717p96_341p02_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_717p96_34p86_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_717p96_400p03_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_717p96_40p51_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_717p96_475p80_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_717p96_47p08_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_717p96_54p71_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_717p96_577p65_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_717p96_63p58_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_717p96_73p89_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_717p96_85p86_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_717p96_99p78_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_800p00_100p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_800p00_200p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_800p00_400p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_800p00_50p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_800p00_700p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_846p11_101p43_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_846p11_118p11_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_846p11_137p54_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_846p11_160p17_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_846p11_186p51_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_846p11_217p19_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_846p11_252p91_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_846p11_294p51_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_846p11_30p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_846p11_345p53_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_846p11_34p93_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_846p11_405p40_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_846p11_40p68_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_846p11_475p64_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_846p11_47p37_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_846p11_558p06_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_846p11_55p16_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_846p11_64p24_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_846p11_654p75_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_846p11_74p80_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_846p11_87p10_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_997p14_101p43_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_997p14_118p11_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_997p14_137p54_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_997p14_160p17_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_997p14_186p51_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_997p14_217p19_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_997p14_254p82_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_997p14_298p97_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_997p14_30p00_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_997p14_34p93_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_997p14_350p77_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_997p14_40p68_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_997p14_411p54_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_997p14_47p37_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_997p14_482p85_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_997p14_55p16_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_997p14_566p51_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_997p14_64p24_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_997p14_664p66_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_997p14_74p80_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_997p14_779p83_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83',
    'HToZATo2L2B_997p14_87p10_v6.1.0+80X_ZAAnalysis_2018-02-16-4-g786cd83'
    #'HToZATo2L2B_MH-200_MA-50_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-200_MA-100_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-250_MA-50_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-250_MA-100_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-300_MA-50_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-300_MA-100_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-300_MA-200_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-500_MA-50_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-500_MA-100_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-500_MA-200_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-500_MA-300_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-500_MA-400_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-650_MA-50_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-800_MA-50_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-800_MA-100_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-800_MA-200_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-800_MA-400_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-800_MA-700_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-1000_MA-50_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-1000_MA-200_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-1000_MA-500_13TeV-madgraph_Summer16MiniAODv2',
    ##'HToZATo2L2B_MH-2000_MA-1000_13TeV-madgraph_Summer16MiniAODv2',
    ##'HToZATo2L2B_MH-3000_MA-2000_13TeV-madgraph_Summer16MiniAODv2'
]

# Number of samples used as basis for the reweighting
number_of_bases = 14

