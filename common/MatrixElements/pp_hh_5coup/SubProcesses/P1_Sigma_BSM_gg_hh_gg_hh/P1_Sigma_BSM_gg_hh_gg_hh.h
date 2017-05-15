// 
// *  This file was automatically generated by MoMEMta-MaGMEE,
// *  A MadGraph Matrix Element Exporter plugin for MoMEMta.
// *
// *  It is subject to MoMEMta-MaGMEE's license and copyright:
// *
// *  Copyright (C) 2016  Universite catholique de Louvain (UCL), Belgium
// *
// *  This program is free software: you can redistribute it and/or modify
// *  it under the terms of the GNU General Public License as published by
// *  the Free Software Foundation, either version 3 of the License, or
// *  (at your option) any later version.
// *
// *  This program is distributed in the hope that it will be useful,
// *  but WITHOUT ANY WARRANTY; without even the implied warranty of
// *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// *  GNU General Public License for more details.
// *
// *  You should have received a copy of the GNU General Public License
// *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
// 

#pragma once

#include <complex> 
#include <vector> 
#include <utility> 
#include <map> 
#include <functional> 

#include <Parameters_BSM_gg_hh.h> 
#include "../../include/SubProcess.h"

#include <MatrixElement.h> 

namespace pp_hh_5coup_BSM_gg_hh 
{

//==========================================================================
// A class for calculating the matrix elements for
// Process: g g > h h WEIGHTED<=6 @1
//--------------------------------------------------------------------------

class P1_Sigma_BSM_gg_hh_gg_hh: public momemta::MatrixElement 
{
  public:

    // Constructor & destructor
    P1_Sigma_BSM_gg_hh_gg_hh(const std::string& param_card); 
    virtual ~P1_Sigma_BSM_gg_hh_gg_hh() {}; 

    // Calculate flavour-independent parts of cross section.
    virtual momemta::MatrixElement::Result compute(
    const std::pair < std::vector<double> , std::vector<double> >
        &initialMomenta,
    const std::vector < std::pair < int, std::vector<double> > > &finalState); 

    virtual std::shared_ptr < momemta::MEParameters > getParameters() 
    {
      return params; 
    }

    // needed? const std::vector<double>& getMasses() const {return mME;}

  private:

    // default constructor should be hidden
    P1_Sigma_BSM_gg_hh_gg_hh() = delete; 

    // list of helicities combinations
    const int helicities[4][4] = {{-1, -1, 0, 0}, {-1, 1, 0, 0}, {1, -1, 0, 0},
        {1, 1, 0, 0}};

    // Private functions to calculate the matrix element for all subprocesses
    // Wavefunctions
    void calculate_wavefunctions(const int perm[], const int hel[]); 
    std::complex<double> amp[2]; 

    // Matrix elements
    double matrix_1_gg_hh(); 

    // map of final states
    std::map < std::vector<int> , std::vector < SubProcess <
        P1_Sigma_BSM_gg_hh_gg_hh >> > mapFinalStates;

    // Reference to the model parameters instance passed in the constructor
    std::shared_ptr < Parameters_BSM_gg_hh > params; 

    // vector with external particle masses
    std::vector < std::reference_wrapper<double> > mME; 

    // vector with momenta (to be changed each event)
    double * momenta[4]; 
}; 


}

