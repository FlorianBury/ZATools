####### Warning: put most recent tags first! ###### 
analysis_tags = [
        'v6.2.0+80X-13-g03d6d79_ZAAnalysis_2018-02-16-4-g786cd83'
        #'v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a'
        #'v6.1.0+80X_ZAAnalysis_2018-02-16-3-gd29729a'  #JEC splitting
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
    'HToZATo2L2B_127p34_30p00',
    'HToZATo2L2B_127p34_34p66',
    'HToZATo2L2B_127p34_40p10',
    'HToZATo2L2B_127p34_46p23',
    'HToZATo2L2B_127p34_53p10',
    'HToZATo2L2B_127p34_60p74',
    'HToZATo2L2B_127p34_69p01',
    'HToZATo2L2B_135p61_30p00',
    'HToZATo2L2B_135p61_34p66',
    'HToZATo2L2B_135p61_40p50',
    'HToZATo2L2B_135p61_47p16',
    'HToZATo2L2B_135p61_54p70',
    'HToZATo2L2B_135p61_63p15',
    'HToZATo2L2B_135p61_72p48',
    'HToZATo2L2B_135p61_81p94',
    'HToZATo2L2B_143p76_30p00',
    'HToZATo2L2B_143p76_34p66',
    'HToZATo2L2B_143p76_40p88',
    'HToZATo2L2B_143p76_48p06',
    'HToZATo2L2B_143p76_56p27',
    'HToZATo2L2B_143p76_65p56',
    'HToZATo2L2B_143p76_75p94',
    'HToZATo2L2B_143p76_86p58',
    'HToZATo2L2B_152p86_30p00',
    'HToZATo2L2B_152p86_34p65',
    'HToZATo2L2B_152p86_40p87',
    'HToZATo2L2B_152p86_48p59',
    'HToZATo2L2B_152p86_57p49',
    'HToZATo2L2B_152p86_67p68',
    'HToZATo2L2B_152p86_79p18',
    'HToZATo2L2B_152p86_91p20',
    'HToZATo2L2B_163p18_30p00',
    'HToZATo2L2B_163p18_34p61',
    'HToZATo2L2B_163p18_40p81',
    'HToZATo2L2B_163p18_49p11',
    'HToZATo2L2B_163p18_58p82',
    'HToZATo2L2B_163p18_70p05',
    'HToZATo2L2B_163p18_82p86',
    'HToZATo2L2B_163p18_96p54',
    'HToZATo2L2B_179p67_104p27',
    'HToZATo2L2B_179p67_120p72',
    'HToZATo2L2B_179p67_30p00',
    'HToZATo2L2B_179p67_34p48',
    'HToZATo2L2B_179p67_40p63',
    'HToZATo2L2B_179p67_49p17',
    'HToZATo2L2B_179p67_60p04',
    'HToZATo2L2B_179p67_72p86',
    'HToZATo2L2B_179p67_87p75',
    'HToZATo2L2B_197p83_108p56',
    'HToZATo2L2B_197p83_128p66',
    'HToZATo2L2B_197p83_146p31',
    'HToZATo2L2B_197p83_30p00',
    'HToZATo2L2B_197p83_34p36',
    'HToZATo2L2B_197p83_40p31',
    'HToZATo2L2B_197p83_48p71',
    'HToZATo2L2B_197p83_60p38',
    'HToZATo2L2B_197p83_74p19',
    'HToZATo2L2B_197p83_90p14',
    'HToZATo2L2B_224p19_107p10',
    'HToZATo2L2B_224p19_129p31',
    'HToZATo2L2B_224p19_154p95',
    'HToZATo2L2B_224p19_174p09',
    'HToZATo2L2B_224p19_30p00',
    'HToZATo2L2B_224p19_34p21',
    'HToZATo2L2B_224p19_39p89',
    'HToZATo2L2B_224p19_47p96',
    'HToZATo2L2B_224p19_59p31',
    'HToZATo2L2B_224p19_72p85',
    'HToZATo2L2B_224p19_88p74',
    'HToZATo2L2B_254p07_105p06',
    'HToZATo2L2B_254p07_127p09',
    'HToZATo2L2B_254p07_153p87',
    'HToZATo2L2B_254p07_186p37',
    'HToZATo2L2B_254p07_30p00',
    'HToZATo2L2B_254p07_34p03',
    'HToZATo2L2B_254p07_39p44',
    'HToZATo2L2B_254p07_47p03',
    'HToZATo2L2B_254p07_58p04',
    'HToZATo2L2B_254p07_71p29',
    'HToZATo2L2B_254p07_86p90',
    'HToZATo2L2B_287p93_103p03',
    'HToZATo2L2B_287p93_124p67',
    'HToZATo2L2B_287p93_150p91',
    'HToZATo2L2B_287p93_182p76',
    'HToZATo2L2B_287p93_221p49',
    'HToZATo2L2B_287p93_34p03',
    'HToZATo2L2B_287p93_39p21',
    'HToZATo2L2B_287p93_46p44',
    'HToZATo2L2B_287p93_57p05',
    'HToZATo2L2B_287p93_69p89',
    'HToZATo2L2B_326p30_112p46',
    'HToZATo2L2B_326p30_135p84',
    'HToZATo2L2B_326p30_164p14',
    'HToZATo2L2B_326p30_198p43',
    'HToZATo2L2B_326p30_36p91',
    'HToZATo2L2B_326p30_42p90',
    'HToZATo2L2B_326p30_51p46',
    'HToZATo2L2B_326p30_63p08',
    'HToZATo2L2B_326p30_76p95',
    'HToZATo2L2B_382p59_115p54',
    'HToZATo2L2B_382p59_138p93',
    'HToZATo2L2B_382p59_167p40',
    'HToZATo2L2B_382p59_201p77',
    'HToZATo2L2B_382p59_242p97',
    'HToZATo2L2B_382p59_293p17',
    'HToZATo2L2B_382p59_30p00',
    'HToZATo2L2B_382p59_44p90',
    'HToZATo2L2B_382p59_53p90',
    'HToZATo2L2B_382p59_65p85',
    'HToZATo2L2B_382p59_79p94',
    'HToZATo2L2B_382p59_96p27',
    'HToZATo2L2B_448p60_111p59',
    'HToZATo2L2B_448p60_133p31',
    'HToZATo2L2B_448p60_159p62',
    'HToZATo2L2B_448p60_191p63',
    'HToZATo2L2B_448p60_230p00',
    'HToZATo2L2B_448p60_275p72',
    'HToZATo2L2B_448p60_30p00',
    'HToZATo2L2B_448p60_331p60',
    'HToZATo2L2B_448p60_44p02',
    'HToZATo2L2B_448p60_53p33',
    'HToZATo2L2B_448p60_64p68',
    'HToZATo2L2B_448p60_78p04',
    'HToZATo2L2B_448p60_93p58',
    'HToZATo2L2B_526p00_111p46',
    'HToZATo2L2B_526p00_132p16',
    'HToZATo2L2B_526p00_157p04',
    'HToZATo2L2B_526p00_187p13',
    'HToZATo2L2B_526p00_223p11',
    'HToZATo2L2B_526p00_266p04',
    'HToZATo2L2B_526p00_30p00',
    'HToZATo2L2B_526p00_318p37',
    'HToZATo2L2B_526p00_36p50',
    'HToZATo2L2B_526p00_383p08',
    'HToZATo2L2B_526p00_44p40',
    'HToZATo2L2B_526p00_456p30',
    'HToZATo2L2B_526p00_54p02',
    'HToZATo2L2B_526p00_65p72',
    'HToZATo2L2B_526p00_79p01',
    'HToZATo2L2B_526p00_94p17',
    'HToZATo2L2B_624p67_124p74',
    'HToZATo2L2B_624p67_146p74',
    'HToZATo2L2B_624p67_173p06',
    'HToZATo2L2B_624p67_204p63',
    'HToZATo2L2B_624p67_242p47',
    'HToZATo2L2B_624p67_288p21',
    'HToZATo2L2B_624p67_30p00',
    'HToZATo2L2B_624p67_343p89',
    'HToZATo2L2B_624p67_35p13',
    'HToZATo2L2B_624p67_413p63',
    'HToZATo2L2B_624p67_41p14',
    'HToZATo2L2B_624p67_48p18',
    'HToZATo2L2B_624p67_505p30',
    'HToZATo2L2B_624p67_56p42',
    'HToZATo2L2B_624p67_66p07',
    'HToZATo2L2B_624p67_77p38',
    'HToZATo2L2B_624p67_90p61',
    'HToZATo2L2B_741p85_108p70',
    'HToZATo2L2B_741p85_128p32',
    'HToZATo2L2B_741p85_178p83',
    'HToZATo2L2B_741p85_215p56',
    'HToZATo2L2B_741p85_264p18',
    'HToZATo2L2B_741p85_30p00',
    'HToZATo2L2B_741p85_330p82',
    'HToZATo2L2B_741p85_35p13',
    'HToZATo2L2B_741p85_41p14',
    'HToZATo2L2B_741p85_426p37',
    'HToZATo2L2B_741p85_48p18',
    'HToZATo2L2B_741p85_519p36',
    'HToZATo2L2B_741p85_56p42',
    'HToZATo2L2B_741p85_644p16',
    'HToZATo2L2B_741p85_66p07',
    'HToZATo2L2B_741p85_78p00',
    'HToZATo2L2B_741p85_92p08',
    'HToZATo2L2B_881p02_113p16',
    'HToZATo2L2B_881p02_133p58',
    'HToZATo2L2B_881p02_157p69',
    'HToZATo2L2B_881p02_186p16',
    'HToZATo2L2B_881p02_219p76',
    'HToZATo2L2B_881p02_259p43',
    'HToZATo2L2B_881p02_30p00',
    'HToZATo2L2B_881p02_35p42',
    'HToZATo2L2B_881p02_382p68',
    'HToZATo2L2B_881p02_41p81',
    'HToZATo2L2B_881p02_49p35',
    'HToZATo2L2B_881p02_533p65',
    'HToZATo2L2B_881p02_58p26',
    'HToZATo2L2B_881p02_644p17',
    'HToZATo2L2B_881p02_68p78',
    'HToZATo2L2B_881p02_81p20',
    'HToZATo2L2B_881p02_95p85'

    #'HToZATo2L2B_MH-1000_MA-500_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-1000_MA-50_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-2000_MA-1000_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-200_MA-100_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-200_MA-50_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-250_MA-100_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-250_MA-50_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-3000_MA-2000_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-300_MA-100_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-300_MA-200_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-300_MA-50_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-500_MA-100_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-500_MA-200_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-500_MA-300_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-500_MA-400_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-500_MA-50_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-650_MA-50_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-800_MA-100_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-800_MA-200_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-800_MA-400_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-800_MA-50_13TeV-madgraph_Summer16MiniAODv2',
    #'HToZATo2L2B_MH-800_MA-700_13TeV-madgraph_Summer16MiniAODv2'
    
]

# Number of samples used as basis for the reweighting
number_of_bases = 14

