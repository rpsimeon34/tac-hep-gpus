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

#define PI 3.14159265

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
Particle::Particle(double energy, double in_pt, double in_eta, double in_phi){
    E = energy;
    pt = in_pt;
    phi = in_phi;
    eta = in_eta;
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
    m = mass;
}

//
//*** Prints 4-vector ----------------------------------------------------------
//
void Particle::print(){
	std::cout << std::endl;
	std::cout << "4-momentum: (" << p[0] <<",\t" << p[1] <<",\t"<< p[2] <<",\t"<< p[3] << ")" << " and sintheta: " <<  sintheta() << std::endl;
    std::cout << "(E,pT,eta,phi): (" << E <<",\t" << pt <<",\t"<< eta <<",\t"<< phi << ")" << " and mass: " << m << std::endl;
}

class Lepton: public Particle {
    public:
    Lepton();
    Lepton(double, double, double, double, double);
    double charge;
    void setCharge(double);
};

Lepton::Lepton() {
    E = pt = eta = phi = m = 0.0;
    p[0] = p[1] = p[2] = p[3] = 0.0;
    setCharge(0.0);
}

Lepton::Lepton(double energy, double in_pt, double in_eta, double in_phi, double in_charge){
    E = energy;
    pt = in_pt;
    phi = in_phi;
    eta = in_eta;
    p4(pt,eta,phi,E);
    double mass_sq = E*E-p[1]*p[1]-p[2]*p[2]-p[3]*p[3];
    if (mass_sq > 0.0) {
        setMass(sqrt(mass_sq));
    } else {
        setMass(0.0);
    }
    setCharge(in_charge);
}

void Lepton::setCharge(double q){
    charge = q;
}

class Jet: public Particle {
    public:
    Jet();
    Jet(double, double, double, double, double);
    double flavor;
    void setFlavor(double);
};

Jet::Jet() {
    E = pt = eta = phi = m = 0.0;
    p[0] = p[1] = p[2] = p[3] = 0.0;
    setFlavor(0.0);
}

Jet::Jet(double energy, double in_pt, double in_eta, double in_phi, double flavor){
    E = energy;
    pt = in_pt;
    phi = in_phi;
    eta = in_eta;
    p4(pt,eta,phi,E);
    double mass_sq = E*E-p[1]*p[1]-p[2]*p[2]-p[3]*p[3];
    if (mass_sq > 0.0) {
        setMass(sqrt(mass_sq));
    } else {
        setMass(0.0);
    }
    setFlavor(flavor);
}

void Jet::setFlavor(double f){
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

    //loop variable
    int n;

	for (Long64_t jentry=0; jentry<100;jentry++)
 	{
		t1->GetEntry(jentry);
		std::cout<<"----------- Event "<< jentry <<" -----------"<<std::endl;


//      Assume lepton array is filled early, so ends at first zero
        std::cout << "LEPTONS:" << std::endl;
        n = 0;
        while (lepE[n]>0) {
            Lepton ev_lep(lepE[n],lepPt[n],lepEta[n],lepPhi[n],lepQ[n]);
            ev_lep.print();
            std::cout << "Charge: " << ev_lep.charge << std::endl;
            n++;
        }
        std::cout << "JETS:" << std::endl;
        for (int m=0; m<njets; m++) {
            Jet ev_jet(jetE[m],jetPt[m],jetEta[m],jetPhi[m],jetHadronFlavour[m]);
            ev_jet.print();
            std::cout << "Hadron flavor: " << ev_jet.flavor << std::endl;
        }


	} // Loop over all events

  	return 0;
}
