from typing import Type

from config import DistributionTypeEnum
from utils.binom import BinomDistributionHandler
from utils.core import BaseDistributionHandler
from utils.nbinom import NBinomDistributionHandler
from utils.pois import PoissonDistributionHandler


class DistributionController:

    @staticmethod
    def get_distribution_handler(distribution_type) -> Type[BaseDistributionHandler]:
        distribution_map = {
            DistributionTypeEnum.BINOM: BinomDistributionHandler,
            DistributionTypeEnum.NBINOM: NBinomDistributionHandler,
            DistributionTypeEnum.POISSON: PoissonDistributionHandler
        }

        return distribution_map[distribution_type]
