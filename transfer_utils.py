import logging

logger = logging.getLogger(__name__)

def transfer_weights(pretrained_model, new_model):
    """Actor, Critic, Critic Target 네트워크의 파라미터를 가능한 범위 내에서 전이"""
    
    def copy_matching_layers(source_net, target_net):
        src_state = source_net.state_dict()
        tgt_state = target_net.state_dict()
        transferred = 0
        total = len(tgt_state)
        new_state = {}

        for k in tgt_state:
            if k in src_state and src_state[k].shape == tgt_state[k].shape:
                new_state[k] = src_state[k]
                transferred += 1
            else:
                new_state[k] = tgt_state[k]  # fallback

        target_net.load_state_dict(new_state)
        return transferred, total

    results = {}

    transferred, total = copy_matching_layers(pretrained_model.actor, new_model.actor)
    results['actor'] = (transferred, total)

    if hasattr(pretrained_model, "critic") and hasattr(new_model, "critic"):
        transferred, total = copy_matching_layers(pretrained_model.critic, new_model.critic)
        results['critic'] = (transferred, total)

    if hasattr(pretrained_model, "critic_target") and hasattr(new_model, "critic_target"):
        transferred, total = copy_matching_layers(pretrained_model.critic_target, new_model.critic_target)
        results['critic_target'] = (transferred, total)

    logger.info(f"🔁 전이 결과: {results}")
    return results
