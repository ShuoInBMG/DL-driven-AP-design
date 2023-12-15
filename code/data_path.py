import os
import shutil
import subprocess

class path:
    def __init__(self, work_path:str, martini_path:str, proc:int)->None:
        if work_path == "default":
            self.work_path = r"D:\LabResearch\assem5\MDsimulation"
        else:
            self.work_path = work_path
        if martini_path == "default":
            self.martini_path = r"D:\LabResearch\assem5\code\martini2\martinize.py"
        else:
            self.martini_path = martini_path
        # self.work_path                                                # D:\YangShuo\CBD-ML\assem5\MDsimulation
        # self.martini_path                                             # D:\YangShuo\CBD-ML\assem5\code\martini2\martinize.py
        self.proc   = proc                                              # 1
        self.folder = self.work_path+f"\{self.proc}"                    # D:\YangShuo\CBD-ML\assem5\MDsimulation\1
        self.pdb    = f"{self.proc}.pdb"                                     # D:\YangShuo\CBD-ML\assem5\MDsimulation\1\1.pdb

class prepare_simulation:
    def __init__(self, parent_fasta:str, parent_pdb:str):
        if parent_fasta is not False:
            self.parent_fasta = parent_fasta                # D:\YangShuo\CBD-ML\assem5\MDsimulation\parent.fasta
            self.parent_pdb = parent_pdb                    # D:\YangShuo\CBD-ML\assem5\MDsimulation\parent

            self.command_0()
    def command_0(self):
        cdr = f"omegafold {self.parent_fasta} {self.parent_pdb}"
        info = os.system(cdr)
        if info == 0:
            print(f"Info omegafold prediction completed at {self.parent_pdb}")
        else:
            pass

class do_simulation:
    def __init__(self, path_object:path, nmol = 2):
        self.path = path_object
        self.nmol = nmol
        os.chdir(self.path.folder)
        # 假设这个文件夹里有一个名为peptide
        self.command_1()
        self.command_2()
        self.command_3()
        self.command_4()
        self.command_5()
    def get_second_number(self, string:str):
        string = string.strip()
        numbers = string.split()
        return float(numbers[1])
    def collect_sasa(self, doc:list):
        sasa = [self.get_second_number(x) for x in doc[24:]]
        init_sasa = sasa[0]
        final_sasa = sum(sasa[960:])/len(sasa[960:])
        return init_sasa / final_sasa
        #return max(sasa) / min(sasa)
    def command_1(self):                # Do dssp calculation.
        # 1. gromacs加H原子，这一步会得到.gro
        cdr = f"gmx pdb2gmx -f {self.path.pdb} \
                -o peptide.pdb \
                -ignh \
                -ff charmm27 \
                -water spc"
        os.system(cdr)
        # 2. 删掉多余的.top和.itp
        os.system("del topol.top")
        os.system("del posre.itp")
        # 3. 算二级结构得到.dssp
        cdr = f"dssp -i peptide.pdb -o peptide.dssp"
        os.system(cdr)

    def command_2(self):                # Do martini cg.
        cdr = f"python {self.path.martini_path} \
                -f peptide.pdb \
                -ss peptide.dssp \
                -o topol.top \
                -x cg.pdb \
                -name peptide \
                -ff martini22"
        os.system(cdr)
    
    def command_3(self):                # Add description to topol.top
        with open("topol.top", "r") as f:
            doc = f.readlines()
        doc[0] = '#include "martini_v2.2.itp"\n'
        doc[1] = '#include "martini_v2.0_ions.itp"\n'
        with open("topol.top", "w") as f:
            f.writelines(doc)
            f.close()
    
    def command_4(self):
        cdr = f"gmx insert-molecules \
               -box 15 15 15 \
               -nmol {self.nmol} \
               -ci cg.pdb \
               -radius 0.4 \
               -o protein_box.gro"
        os.system(cdr)
        
    def command_5(self):
        shutil.copyfile(r"D:\LabResearch\assem5\code\mdpFile\water.gro",
                        "water.gro")
        cdr = "gmx solvate \
               -cp protein_box.gro \
               -cs water.gro \
               -p topol.top \
               -o protein_sol.gro \
               -radius 0.25"
        os.system(cdr)
        with open("topol.top","r") as f:
            doc = f.readlines()
        string = doc[-1]
        clean_string = string.replace('\t', ' ').replace('\n', ' ')
        split_string = clean_string.split()
        result = [s.strip() for s in split_string]
        new_doc = doc[:-1]
        new_doc.append(result[0]+"\t"+f"{self.nmol}\n")
        new_doc.append("W"+"\t"+result[-1]+"\n")
        with open("topol.top","w") as f:
            f.writelines(new_doc)
        f.close()
    
    def command_6(self):
        shutil.copyfile(r"D:\LabResearch\assem5\code\mdpFile\em.mdp",
                        "em.mdp")
        cdr = "gmx grompp -f em.mdp -c protein_sol.gro -p topol.top -o em -maxwarn 1"
        os.system(cdr)
        cdr = "gmx genion -s em.tpr -p topol.top -o protein_sol.gro -pname NA+ -nname CL- -neutral"
        process = subprocess.Popen(cdr, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate(input=b"13")
        shutil.copyfile(r"D:\LabResearch\assem5\code\mdpFile\minim.mdp",
                         "minim.mdp")
        cdr = "gmx grompp -f em.mdp -c protein_sol.gro -p topol.top -o em -maxwarn 2"
        os.system(cdr)
        cdr = "gmx mdrun -deffnm em"
        os.system(cdr)
        os.system("cls")
    
    def command_7(self):
        shutil.copyfile(r"D:\LabResearch\assem5\code\mdpFile\md.mdp",
                         "md.mdp")
        cdr = "gmx grompp -f md.mdp -c em.gro -p topol.top -o md -maxwarn 2"
        os.system(cdr)
        cdr = "gmx mdrun -v -deffnm md"
        os.system(cdr)
    
    def command_8(self):
        cdr = "gmx sasa -s md.tpr -f md.xtc -o area.xvg"
        process = subprocess.Popen(cdr, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate(input=b"1")

        with open("area.xvg","r") as f:
            doc = f.readlines()
            f.close()
        ap = self.collect_sasa(doc)
        print(f"AP: {ap:.4f}")
        # return ap
        