#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is an example code to be included as supplementary material to the following article: 
"AutoHood3D: A Multi-Modal Benchmark for Automotive Hood Design and Fluid–Structure Interaction".

This is a demonstration intended to provide a working example with data provided in the repo.
For application on other datasets, the requirement is to configure the settings. 

Dependencies: 
    - Python package list provided: package.list

Running the code: 
    - assuming above dependencies are configured, "python <this_file.py>" will run the demo code. 
    NOTE - It is important to check README prior to running this code.

Contact: 
    - Vansh Sharma at vanshs@umich.edu
    - Harish Jai Ganesh at harishjg@umich.edu
    - Venkat Raman at ramanvr@umich.edu

Affiliation: 
    - APCL Group 
    - Dept. of Aerospace Engg., University of Michigan, Ann Arbor
"""

import os
import shutil
import subprocess
import time

def run_command(command, cwd=None, log_file=None, check=True, background=False):
    if background:
        return subprocess.Popen(command, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
    else:
        result = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        if log_file:
            with open(log_file, 'wb') as f:
                f.write(result.stdout + result.stderr)
        if check and result.returncode != 0:
            raise RuntimeError(f"Command failed: {command}\n{result.stderr.decode()}")
        return result

def run_hood(hood_dir, hood_name, run_dir, solution_dir, num_procs):
    original_hood_name = os.path.splitext(hood_name)[0]
    hood_path = os.path.join(hood_dir, hood_name)

    log_dir = os.path.join(solution_dir, original_hood_name)
    shutil.rmtree(log_dir, ignore_errors=True)
    os.makedirs(log_dir)

    # Remove old precice-run directory if exists
    shutil.rmtree(os.path.join(run_dir, 'precice-run'), ignore_errors=True)


    all_log = os.path.join(log_dir, 'all.log')

    fluid_dir = os.path.join(run_dir, 'fluid_2_thickShell')
    solid_dir = os.path.join(run_dir, 'solid')
    trisurface_fluid_dir = os.path.join(fluid_dir, 'constant/triSurface')
    trisurface_solid_dir = os.path.join(solid_dir, 'constant/triSurface')

    shutil.rmtree(trisurface_fluid_dir, ignore_errors=True)
    shutil.rmtree(trisurface_solid_dir, ignore_errors=True)
    os.makedirs(trisurface_fluid_dir)
    os.makedirs(trisurface_solid_dir)

    new_hood_name = 'hood_with_2_combined_shapes_iter_1.stl'
    shutil.copy(hood_path, os.path.join(trisurface_fluid_dir, new_hood_name))
    shutil.copy(hood_path, os.path.join(trisurface_solid_dir, new_hood_name))

    try:
        hood_id = int(original_hood_name[4:7])
    except Exception as e:
        with open(all_log, "a") as log:
            log.write(f"Invalid hood name: {hood_name}\n")
        return

    location_dict = {
            1 : '(0.5479223425033738 0.7950395532761292 0.006321841367240177);',
            2 : '(0.4820818387684196 0.7587534129765782 0.007523938636067040);',
            3 : '(0.47365486122903167 0.7652157580633456 -0.006150654349216515);',
            4 : '(0.4656999658419894 -0.8013127678433543 0.025949855850267267);',
            5 : '(0.5728536234993133 0.7383375821642415 0.013970549338926247);',
            6 : '(0.5225935762237234 -0.7665625656576996 0.0015983277109045818);',
            7 : '(0.5798244176690328 -0.8093402563459539 -0.056616032822216555);',
            8 : '(0.4791706206337554 -0.7982623220796783 -0.05050354412091626);',
            9 : '(0.482885128315683 -0.8092237885360021 -0.04572764068755852);',
            10 : '(0.5784247144727567 -0.7982625533056066 -0.046958193478192987);',
            11 : '(0.5726378672660771 0.738337563591182 0.01601400859891698);',
            12 : '(0.5474496193641774 0.795119913412087 0.006986560781734971);',
            13 : '(0.547117948000667 0.795097590815898 0.00848655634681697);',
            14 : '(0.5786680148662388 -0.7982626377830733 -0.04644515916051308);',
            15 : '(0.5785513638315065 -0.798262546620356 -0.046951087431538626);',
            16 : '(0.5473360330412536 0.7950974825802293 0.007849256931639514);',
            17 : '(0.547110719185429 0.7950972537735649 0.008996856495098224);',
            18 : '(0.547093549030197 0.7950970996881306 0.009166113704282844);',
            19 : '(0.547120884505103 0.7950972301577557 0.008875381477654778);',
            20 : '(0.5471381554958211 0.795097149265631 0.008633376783492505);',
            21 : '(0.5785056794596514 -0.7982625460668423 -0.04596092046820393);',
            22 : '(0.5785593844740083 -0.79826255079954 -0.04719267931272326);',
            23 : '(0.4817089262201794 0.7587537090340499 0.014534410762405539);',
            24 : '(0.5785753471343066 -0.7982625450562625 -0.04821435862114671);',
            25 : '(0.481648898536911 0.7587471562955154 0.014847090166694818);',
            26 : '(0.481766527734609 0.7587537016041266 0.01409098861420012);',
            27 : '(0.4817051831213479 0.7587537150546286 0.014559620810047844);',
            28 : '(0.481868008977012 0.7587537358895948 0.013588333810507836);',
            29 : '(0.4817480298366471 0.7587536507002115 0.014186199949112271);',
            30 : '(0.4816084208395784 0.7587471842252613 0.0156357070574477);',
            31 : '(0.4816681389792475 0.7587471480099963 0.014766399681636774);',
            32 : '(0.473852046622511 0.7652089895612273 -0.010496295726164534);',
            33 : '(0.4738780539998548 0.7652159050653703 -0.012639252353712757);',
            34 : '(0.4738284689250966 0.7652158300651409 -0.012782801329820036);',
            35 : '(0.4737847711848123 0.7652157584407143 -0.009743032501163137);',
            36 : '(0.4737634420491135 0.7652157571344617 -0.00855207048171433);',
            37 : '(0.4737715675689419 0.7652157356906999 -0.009009917985932904);',
            38 : '(0.4737348024028282 0.7652155583566888 -0.006901473804779561);',
            39 : '(0.4737280441387383 0.76520895871403 -0.008048241652929361);',
            40 : '(0.4737023729746752 0.7652153328287283 -0.007020602829894494);',
            41 : '(0.4738016823328794 0.7652157709248937 -0.009796873110837247);',
            42 : '(0.4648314647832495 -0.8013123118895458 0.027667387364502166);',
            43 : '(0.46444758475489176 -0.8013123358519262 0.02947341354092614);',
            44 : '(0.4642666578072416 -0.8013120063404351 0.03063549785472418);',
            45 : '(0.4643230633354614 -0.8013124226053904 0.030234079059829125);',
            46 : '(0.46435180507972457 -0.8013206146533858 0.03038787467498999);',
            47 : '(0.4643092468837849 -0.8013158150171162 0.03034412452069135);',
            48 : '(0.46426379224715075 -0.8013169739792756 0.03134122522948062);',
            49 : '(0.4642608221182617 -0.8013124154517683 0.031124904949128424);',
            50 : '(0.46427219051433566 -0.8013124210660885 0.031137328833242425);',
            51 : '(0.4644265252626678 -0.8013124707082675 0.02982436026300039);',
            52 : '(0.5727400826818856 0.738337582367823 0.0141594339945626);',
            53 : '(0.5727252951823818 0.7383376272885734 0.01417989972895617);',
            54 : '(0.5725989504708179 0.738337489759841 0.01686084935513461);',
            55 : '(0.572655320504804 0.7383370254062618 0.01294811714524179);',
            56 : '(0.5726957119942331 0.7383374599480518 0.015549151526939114);',
            57 : '(0.5728226605951808 0.738337541005147 0.01523331738676688);',
            59 : '(0.5725682362783612 0.7383375949233881 0.016452543029713065);',
            60 : '(0.572554041100921 0.7383377759838523 0.017198129923062577);',
            61 : '(0.5222954858766521 0.7665637987410115 0.00311273280182425);',
            62 : '(0.5219998486136548 0.7665638077274767 0.005246224537722555);',
            63 : '(0.5218236479468666 0.7665684065393382 0.006290360796262079);',
            64 : '(0.5223325544413342 0.7665638025026459 0.0031846418782689317);',
            65 : '(0.521819484551941 0.7665638109634771 0.00689299607355251);',
            66 : '(0.5217520855569173 0.7665638138067402 0.007647523010616025);',
            67 : '(0.52175838760248 -0.7665580030509667 0.0070595454604845675);',
            68 : '(0.521806218547359 0.7665638142529074 0.007256605830843935);',
            69 : '(0.5217676923181847 0.7665638127928857 0.0075535049302453555);',
            70 : '(0.5218215156360947 0.7665638115574304 0.006888154997720705);',
            71 : '(0.5796891124883952 -0.8093402121249355 -0.05880122599301568);',
            72 : '(0.5797061150184316 -0.8093400962017686 -0.05933720271149837);',
            73 : '(0.5797308744675352 -0.8093402951562322 -0.05725786498864741);',
            74 : '(0.5795332653453528 -0.8093401831287745 -0.06104834340558252);',
            75 : '(0.5798438083009498 -0.809340273781791 -0.05635817792175958);',
            76 : '(0.5797806430327386 -0.809340233245777 -0.05754357994623452);',
            77 : '(0.5798199724987816 -0.8093402810419715 -0.05596513383107726);',
            78 : '(0.5798621107616302 -0.8093402671046795 -0.05645696945380392);',
            79 : '(0.5797783903957547 -0.8093402460549247 -0.05699078520653557);',
            80 : '(0.5798441643615185 -0.8093402718268525 -0.0563516023457452);',
            81 : '(0.4794359582828323 -0.7982624431466899 -0.05827316974526663);',
            82 : '(0.4791010031418884 -0.7982623163347642 -0.05473459072694931);',
            83 : '(0.479162099151681 -0.7982621582417548 -0.05113262001247058);',
            84 : '(0.47928126506154856 -0.798262379098343 -0.05758600475958746);',
            85 : '(0.4791132127262286 -0.7982623155251789 -0.05172450611179977);',
            86 : '(0.4793073834639293 -0.7982623170624749 -0.05594447730340261);',
            87 : '(0.4791810627681818 -0.7982623151210795 -0.05134032019665341);',
            88 : '(0.4791635453069485 -0.7982618963734085 -0.05144837365119027);',
            89 : '(0.4791454207040188 -0.7982623061249663 -0.05026412172538859);',
            90 : '(0.4791296590010785 -0.7982623154996814 -0.05176399977684546);',
            91 : '(0.4828676580298661 -0.8092237236239982 -0.04738104971667381);',
            92 : '(0.4828117768532665 -0.8092237699049609 -0.0472961171703748);',
            93 : '(0.4828242477941259 -0.8092238646399608 -0.04462678940302069);',
            94 : '(0.482907591182432 -0.809223768503488 -0.04956568646592514);',
            95 : '(0.482806711545382 -0.8092238246669063 -0.04471755128500029);',
            96 : '(0.4828454712779057 -0.8092237663969118 -0.04745378427063013);',
            97 : '(0.4827296828681024 -0.809223790828844 -0.04640008655052558);',
            98 : '(0.4827845682003543 -0.8092238291333804 -0.04478523578615071);',
            99 : '(0.4828000981092002 -0.8092238412373195 -0.04417838297298723);',
            100 : '(0.4828068831288816 -0.809223829739204 -0.0447133803540723);',
            101 : '(0.5784754839495443 -0.7982625603342457 -0.05031557071574388);',
            102 : '(0.5784256559910187 -0.7982624324088056 -0.05065402684283818);',
            103 : '(0.5784208078891599 -0.7982619125563963 -0.04748762235054164);',
            104 : '(0.5783582024958003 -0.7982625649717279 -0.05278222806987921);',
            105 : '(0.5785347822880735 -0.7982625462529759 -0.04691090296351434);',
            106 : '(0.5477421451708036 0.795049699571895 0.006657567386315483);',
            107 : '(0.54710703073221 0.7951184754978469 0.008887148121108787);',
            108 : '(0.481982630141995 0.7587536190167753 0.01140922463264954);',
            109 : '(0.48205240 0.75874712 0.01198275);'
    }

    try:
        for case_dir in [fluid_dir, solid_dir]:
            shutil.rmtree(os.path.join(case_dir, '0'), ignore_errors=True)
            shutil.copytree(os.path.join(case_dir, 'master_0'), os.path.join(case_dir, '0'))

        snappy_dict_path = os.path.join(solid_dir, 'system/snappyHexMeshDict')
        with open(snappy_dict_path, 'r') as f:
            content = f.readlines()

        for i, line in enumerate(content):
            if 'locationInMesh' in line and '(' in line:
                content[i] = f'    locationInMesh {location_dict[hood_id]}\n'
                break

        with open(snappy_dict_path, 'w') as f:
            f.writelines(content)

        run_command('surfaceFeatureExtract', cwd=fluid_dir)
        run_command('surfaceFeatureExtract', cwd=solid_dir)
        run_command('blockMesh', cwd=fluid_dir)
        run_command('blockMesh', cwd=solid_dir)
        fluidsnappy_log = open(os.path.join(log_dir, 'fluidSnappy.log'), 'w')
        solidsnappy_log = open(os.path.join(log_dir, 'solidSnappy.log'), 'w')
        solid_snappy = subprocess.Popen('snappyHexMesh -overwrite', cwd=solid_dir, stdout=solidsnappy_log, stderr=subprocess.STDOUT, shell=True)
        fluid_snappy = subprocess.Popen('snappyHexMesh -overwrite', cwd=fluid_dir, stdout=fluidsnappy_log, stderr=subprocess.STDOUT, shell=True)
        solid_snappy.wait()
        fluid_snappy.wait()
        run_command('decomposePar -force', cwd=fluid_dir, log_file=os.path.join(log_dir, 'fluidDecompose.log'))
        run_command('decomposePar -force', cwd=solid_dir, log_file=os.path.join(log_dir, 'solidDecompose.log'))
    except Exception as e:
        with open(all_log, "a") as log:
            log.write(f"Meshing error for {original_hood_name}: {e}\n")
            shutil.copyfile(os.path.join(solid_dir, "system/snappyHexMeshDict"), os.path.join(log_dir, "solidSnappyHexMeshDict"))
        return

    # CheckMesh
    try:
        run_command('checkMesh', cwd=fluid_dir, log_file=os.path.join(log_dir, 'checkMeshFluid.log'))
        run_command('checkMesh', cwd=solid_dir, log_file=os.path.join(log_dir, 'checkMeshSolid.log'))
    except Exception as e:
        with open(all_log, "a") as log:
            log.write(f"CheckMesh failed for {original_hood_name}: {e}\n")

    # Run solvers parallel
    try:
        fluid_command = f'mpirun -np {num_procs} UM_pimpleFoam -parallel'
        solid_command = f'mpirun -np {num_procs} UM_solidDisplacementFoam -parallel'

        fluid_log = open(os.path.join(log_dir, 'pimpleFoam.log'), 'w')
        solid_log = open(os.path.join(log_dir, 'solidDisplacementFoam.log'), 'w')

        fluid_proc = subprocess.Popen(fluid_command, cwd=fluid_dir, stdout=fluid_log, stderr=subprocess.STDOUT, shell=True)
        solid_proc = subprocess.Popen(solid_command, cwd=solid_dir, stdout=solid_log, stderr=subprocess.STDOUT, shell=True)

        fluid_code = fluid_proc.wait()
        solid_code = solid_proc.wait()

        if fluid_code != 0:
            raise RuntimeError(f"Fluid solver crash for {original_hood_name}")
        if solid_code != 0:
            raise RuntimeError(f"Solid solver crash for {original_hood_name}")

    except Exception as e:
        with open(all_log, "a") as log:
            log.write(f"Solver error for {original_hood_name}: {e}\n")
        return

    # Move post-processing results
    try:
        postproc_fluid_dir = os.path.join(fluid_dir, 'postProcessing/sample')
        postproc_solid_dir = os.path.join(solid_dir, 'postProcessing/sample')
        solid_solution = os.path.join(log_dir, 'solid')
        fluid_solution = os.path.join(log_dir, 'fluid')
        os.makedirs(solid_solution, exist_ok=True)
        os.makedirs(fluid_solution, exist_ok=True)

        for source_dir, target_dir, pattern in [
            (postproc_fluid_dir, fluid_solution, 'hoodFluid.vtp'),
            (postproc_solid_dir, solid_solution, 'hoodSolid.vtp')
        ]:
            for time_dir in sorted(os.listdir(source_dir)):
                full_time_dir = os.path.join(source_dir, time_dir)
                if os.path.isdir(full_time_dir):
                    time_suffix = time_dir.replace('0.', '')
                    src = os.path.join(full_time_dir, pattern)
                    if os.path.exists(src):
                        dest = os.path.join(target_dir, f"{pattern.split('.')[0]}_{time_suffix}.vtp")
                        shutil.move(src, dest)

    except Exception as e:
        with open(all_log, "a") as log:
            log.write(f"Post-processing error for {original_hood_name}: {e}\n")

if __name__ == "__main__":
    hood_dir = '/path/to/hoods_dir'
    run_dir = '/path/to/run_dir_1'
    solution_dir = '/path/to/solution/directory'
    num_procs = 12

    list_file = os.path.join(run_dir, 'items_1.txt')
    with open(list_file, 'r') as f:
        hoods = f.read().splitlines()

    for hood_name in hoods:
        run_hood(hood_dir, hood_name, run_dir, solution_dir, num_procs)
