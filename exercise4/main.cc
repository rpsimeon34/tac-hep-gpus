#include <stdio.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "hh/t1.h"

#include <TMath.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TCanvas.h> 
#include <TLorentzVector.h>



//------------------------------------------------------------------------------
// Particle Class
//
class Particle{

	public:
	Particle();
	Particle(double, double, double, double);
	double   pt, eta, phi, E, m, p[4];
	void     p4(double, double, double, double);
	void     print();
	void     setMass(double);
	double   sintheta();
};

//------------------------------------------------------------------------------

//*****************************************************************************
//                                                                             *
//    MEMBERS functions of the Particle Class                                  *
//                                                                             *
//*****************************************************************************

//
//*** Default constructor ------------------------------------------------------
//
Particle::Particle(){
	pt = eta = phi = E = m = 0.0;
	p[0] = p[1] = p[2] = p[3] = 0.0;
}

//*** Additional constructor ------------------------------------------------------
Particle::Particle(double E, double pt, double eta, double phi){
    E = E;
    pt = pt;
    phi = phi;
    eta = eta;
    p4(pt,eta,phi,E);
    setMass(E*E-p[1]*p[1]-p[2]*p[2]-p[3]*p[3]);
}

//
//*** Members  ------------------------------------------------------
//
double Particle::sintheta(){

    double expeta;
    expeta = exp(-1.0*eta);
    return 2.0*expeta/(1.0+expeta*expeta);
}

void Particle::p4(double pT, double eta, double phi, double energy){

    p[1] = pT*cos(phi*PI/180.0);
    p[2] = pT*sin(phi*PI/180.0);
    p[3] = pT*sinh(eta);
    p[0] = energy;

}

void Particle::setMass(double mass)
{
    double m = mass;
}

//
//*** Prints 4-vector ----------------------------------------------------------
//
void Particle::print(){
	std::cout << std::endl;
	std::cout << "(" << p[0] <<",\t" << p[1] <<",\t"<< p[2] <<",\t"<< p[3] << ")" << "  " <<  sintheta() << std::endl;
}

class Lepton: public Particle {
    public:
    double charge;
    void setCharge(double);
}

void Lepton::setCharge(double q){
    charge = q;
}

class Jet: public Particle {
    public:
    double flavor;
    void setFlavor(int);
}

void Jet::setFlavor(int f){
    flavor = f;
}

int main() {
	
	/* ************* */
	/* Input Tree   */
	/* ************* */

	TFile *f      = new TFile("input.root","READ");
	TTree *t1 = (TTree*)(f->Get("t1"));

	// Read the variables from the ROOT tree branches
	t1->SetBranchAddress("lepPt",&lepPt);
	t1->SetBranchAddress("lepEta",&lepEta);
	t1->SetBranchAddress("lepPhi",&lepPhi);
	t1->SetBranchAddress("lepE",&lepE);
	t1->SetBranchAddress("lepQ",&lepQ);
	
	t1->SetBranchAddress("njets",&njets);
	t1->SetBranchAddress("jetPt",&jetPt);
	t1->SetBranchAddress("jetEta",&jetEta);
	t1->SetBranchAddress("jetPhi",&jetPhi);
	t1->SetBranchAddress("jetE", &jetE);
	t1->SetBranchAddress("jetHadronFlavour",&jetHadronFlavour);

	// Total number of events in ROOT tree
	Long64_t nentries = t1->GetEntries();

//	for (Long64_t jentry=0; jentry<100;jentry++)
    for (Long64_t jentry=0; jentry<3;jentry++)
 	{
		t1->GetEntry(jentry);
		std::cout<<" Event "<< jentry <<std::endl;	

		std::cout << sizeof(jetPt) << endl;


	} // Loop over all events

  	return 0;
}
