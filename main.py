from srcTopDown.material_models import ElasticPK2E
import srcTopDown.test.Trial2D_elastoplastic.fixed_disp_runmain as Trial2D_elastoplastic

def main():
    Trial2D_elastoplastic.fixed_disp(plot=True, save=True)
    
if __name__ == "__main__":
    main()