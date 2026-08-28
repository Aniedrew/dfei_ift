from particle import Particle


def get_ref_signal(ref_signal):  # Here we can define them all
    if "inclusive" in ref_signal:
        return {}
    elif 'Bs_Jpsiphi' in ref_signal:
        signal_decay = {'daughters': ['mu+', 'mu-', 'K+', 'K-'], 'mothers': ['B(s)0']}
        cc_signal_decay = {'daughters': ['mu+', 'mu-', 'K+', 'K-'], 'mothers': ['B(s)~0']}
        return signal_decay, cc_signal_decay
    elif "Bd_JpsiKst" in ref_signal or "BdToJpsiKst" in ref_signal:
        signal_decay = {'daughters': ['mu+', 'mu-', 'K+', 'pi-'], 'mothers': ['B0']}
        cc_signal_decay = {'daughters': ['mu+', 'mu-', 'pi+', 'K-'], 'mothers': ['B~0']}
        return signal_decay, cc_signal_decay
    elif 'Bd_JpsiKs' in ref_signal:
        signal_decay = {'daughters': ['mu+', 'mu-', 'pi+', 'pi-'], 'mothers': ['B0']}
        cc_signal_decay = {'daughters': ['mu+', 'mu-', 'pi+', 'pi-'], 'mothers': ['B~0']}
        return signal_decay, cc_signal_decay
    elif "Bs_Dspi" in ref_signal:
        signal_decay = {'daughters': ['K+', 'K-', 'pi+', 'pi-'], 'mothers': ['B(s)0']}
        cc_signal_decay = {'daughters': ['K+', 'K-', 'pi+', 'pi-'], 'mothers': ['B(s)~0']}
        return signal_decay, cc_signal_decay
    elif "Bs_Kmunu" in ref_signal:
        signal_decay = {'daughters': ['K-', 'mu+'], 'mothers': ['B(s)0']}
        cc_signal_decay = {'daughters': ['K+', 'mu-'], 'mothers': ['B(s)~0']}
        return signal_decay, cc_signal_decay
    elif "Bu_JpsiK" in ref_signal or "BuToJpsiK" in ref_signal:
        signal_decay = {'daughters': ['mu+', 'mu-', 'K+'], 'mothers': ['B+']}
        cc_signal_decay = {'daughters': ['mu+', 'mu-', 'K-'], 'mothers': ['B-']}
        return signal_decay, cc_signal_decay
    elif "Bc_Jpsitaunu" in ref_signal or "Bc_Jpsimunu" in ref_signal:
        signal_decay = {'daughters': ['mu+', 'mu-', 'mu+'], 'mothers': ['B(c)+']}
        cc_signal_decay = {'daughters': ['mu+', 'mu-', 'mu-'], 'mothers': ['B(c)-']}
        return signal_decay, cc_signal_decay


    raise NotImplementedError


def particle_name(id_):
    if id_ == 0:
        return 'ghost'
    elif id_ == 10413:
        return 'D1(2420)+'
    elif id_ == -10413:
        return 'D1(2420)-'
    elif id_ == 4412:
        return 'Sigma_cc+'
    elif id_ == -4412:
        return 'Sigma_cc-'
    elif id_ == 4422:
        return 'Chi_cc++'
    elif id_ == -4422:
        return 'Chi_cc--'
    elif id_ == 4432:
        return 'Omega_cc++'
    elif id_ == -4432:
        return 'Omega_cc--'
    else:
        try:
            name = Particle.from_pdgid(id_).name
        except:
            print(id_)
            name = str(id_)
        return name
